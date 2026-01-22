import argparse
import os
import sys
from datetime import datetime
import mediapipe as mp
import numpy as np
import cv2
import torch
from skimage.transform import resize
from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKL, AutoencoderTiny
from src.scheduler.scheduler_ddim import DDIMScheduler
import random
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline, Pose2VideoPipeline_Stream
from src.utils.util import save_videos_grid, crop_face
from decord import VideoReader
from diffusers.utils.import_utils import is_xformers_available

from src.models.motion_encoder.encoder import MotEncoder
from src.liveportrait.motion_extractor import MotionExtractor
from src.models.pose_guider import PoseGuider
from tqdm import tqdm
from huggingface_hub import snapshot_download
import subprocess
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/prompts/personalive_offline.yaml')
    parser.add_argument("--name", type=str, default='personalive_offline_chunked')
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=100, help='Number of frames to process at once (default: 100)')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_xformers", type=bool, default=True)
    parser.add_argument("--stream_gen", type=bool, default=True, help='use streaming generation strategy to reduce VRAM usage.')
    parser.add_argument("--reference_image", type=str, default='', help='Path to reference image. If provided, overrides test_cases from config.')
    parser.add_argument("--driving_video", type=str, default='', help='Path to driving video. If provided, overrides test_cases from config.')
    parser.add_argument("--model_dir", type=str, default='', help='Path to model directory containing personalive subfolder with all model weights. If provided, will auto-detect and use models from this directory.')
    args = parser.parse_args()

    return args

def concatenate_videos(video_paths, output_path, audio_source=None):
    """Concatenate multiple video files using ffmpeg"""
    if len(video_paths) == 0:
        print("No videos to concatenate")
        return False
    
    if len(video_paths) == 1:
        # If only one video, just copy it
        shutil.copy(video_paths[0], output_path)
        if audio_source:
            add_audio_to_video(output_path, audio_source)
        return True
    
    # Create a temporary file list for ffmpeg
    list_file = output_path.replace('.mp4', '_filelist.txt')
    with open(list_file, 'w') as f:
        for video_path in video_paths:
            # Use forward slashes and escape special characters
            escaped_path = video_path.replace('\\', '/').replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    
    try:
        # Use ffmpeg to concatenate
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            # Try alternative method with re-encoding
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', list_file,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
        # Clean up list file
        if os.path.exists(list_file):
            os.remove(list_file)
        
        # Add audio if source is provided
        if audio_source and os.path.exists(output_path):
            add_audio_to_video(output_path, audio_source)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error concatenating videos: {e}")
        if os.path.exists(list_file):
            os.remove(list_file)
        return False

def add_audio_to_video(video_path, audio_source):
    """Add audio from source video to output video"""
    temp_path = video_path.replace('.mp4', '_temp.mp4')
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_source,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            temp_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(temp_path):
            os.replace(temp_path, video_path)
            return True
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    except Exception as e:
        print(f"Warning: Could not add audio: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def main(args):
    device = args.device
    print('device', device)
    config = OmegaConf.load(args.config)
    
    # Auto-detect and override model paths if model_dir is provided
    if args.model_dir:
        model_dir = os.path.abspath(args.model_dir)
        print(f'Using model directory: {model_dir}')
        
        # Check for personalive subfolder
        personalive_dir = os.path.join(model_dir, 'personalive')
        if os.path.exists(personalive_dir):
            # Auto-detect all model files
            denoising_unet_path = os.path.join(personalive_dir, 'denoising_unet.pth')
            reference_unet_path = os.path.join(personalive_dir, 'reference_unet.pth')
            motion_encoder_path = os.path.join(personalive_dir, 'motion_encoder.pth')
            pose_guider_path = os.path.join(personalive_dir, 'pose_guider.pth')
            temporal_module_path = os.path.join(personalive_dir, 'temporal_module.pth')
            motion_extractor_path = os.path.join(personalive_dir, 'motion_extractor.pth')
            
            # Verify all required files exist
            required_files = {
                'denoising_unet': denoising_unet_path,
                'reference_unet': reference_unet_path,
                'motion_encoder': motion_encoder_path,
                'pose_guider': pose_guider_path,
                'temporal_module': temporal_module_path,
                'motion_extractor': motion_extractor_path
            }
            
            missing_files = []
            for name, path in required_files.items():
                if not os.path.exists(path):
                    missing_files.append(f"{name} ({path})")
            
            if missing_files:
                print(f"ERROR: Missing required model files:")
                for missing in missing_files:
                    print(f"  - {missing}")
                sys.exit(1)
            
            # Override config with detected paths
            config.denoising_unet_path = denoising_unet_path
            print(f'Auto-detected models from: {personalive_dir}')
            print(f'  - denoising_unet: {os.path.basename(denoising_unet_path)}')
            print(f'  - reference_unet: {os.path.basename(reference_unet_path)}')
            print(f'  - motion_encoder: {os.path.basename(motion_encoder_path)}')
            print(f'  - pose_guider: {os.path.basename(pose_guider_path)}')
            print(f'  - temporal_module: {os.path.basename(temporal_module_path)}')
            print(f'  - motion_extractor: {os.path.basename(motion_extractor_path)}')
            
            # Check if VAE and base model exist in model_dir, otherwise use HuggingFace
            vae_dir = os.path.join(model_dir, 'sd-vae-ft-mse')
            base_model_dir = os.path.join(model_dir, 'sd-image-variations-diffusers')
            
            if os.path.exists(vae_dir):
                config.vae_path = vae_dir
                print(f'Using VAE from: {vae_dir}')
            else:
                config.vae_path = 'stabilityai/sd-vae-ft-mse'
                print(f'VAE not found in model_dir, downloading from HuggingFace: {config.vae_path}')
            
            if os.path.exists(base_model_dir):
                config.pretrained_base_model_path = base_model_dir
                config.image_encoder_path = os.path.join(base_model_dir, 'image_encoder')
                print(f'Using base model from: {base_model_dir}')
            else:
                config.pretrained_base_model_path = 'lambdalabs/sd-image-variations-diffusers'
                config.image_encoder_path = 'lambdalabs/sd-image-variations-diffusers'
                print(f'Base model not found in model_dir, downloading from HuggingFace: {config.pretrained_base_model_path}')
        else:
            print(f"ERROR: 'personalive' subfolder not found in {model_dir}")
            print(f"Expected structure: {model_dir}/personalive/[model files]")
            sys.exit(1)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(config.vae_path).to(device, dtype=weight_dtype)
    # if use tiny VAE
    # vae_tiny = AutoencoderTiny.from_pretrained(config.vae_tiny_path).to(device, dtype=weight_dtype)

    infer_config = OmegaConf.load(config.inference_config)
    
    # Load reference_unet first to get the cached model path for custom loaders
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device=device, dtype=weight_dtype)
    
    # If using HuggingFace model, get the local cache path for custom loaders
    base_model_path = config.pretrained_base_model_path
    if not os.path.exists(base_model_path):
        # It's a HuggingFace model ID, get the cached path
        base_model_path = snapshot_download(repo_id=config.pretrained_base_model_path)
        print(f'Downloaded base model to cache: {base_model_path}')
        # Update image_encoder_path to use the cached local path
        config.image_encoder_path = os.path.join(base_model_path, 'image_encoder')
    
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device).eval()
    pose_guider = PoseGuider().to(device=device, dtype=weight_dtype)
    pose_encoder = MotionExtractor(num_kp=21).to(device=device, dtype=weight_dtype).eval()
    
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(
        OmegaConf.load(config.inference_config).noise_scheduler_kwargs
    )
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)
    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False
    )
    reference_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'reference_unet'),
            map_location="cpu",
        ),
        strict=True,
    )
    motion_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_encoder'),
            map_location="cpu",
        ),
        strict=True,
    )
    pose_guider.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'pose_guider'),
            map_location="cpu",
        ),
        strict=True,
    )
    denoising_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'temporal_module'),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_extractor'),
            map_location="cpu",
        ),
        strict=False,
    )
    
    if args.use_xformers:
        if is_xformers_available(): 
            try:
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print("Failed to enable xformers:", e)
        else:
            print("xformers is not available. Make sure it is installed correctly.")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    if args.stream_gen:
        pipeline = Pose2VideoPipeline_Stream
    else:
        pipeline = Pose2VideoPipeline
    
    pipe = pipeline(
        vae=vae,
        # vae_tiny=vae_tiny,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        motion_encoder=motion_encoder,
        pose_encoder=pose_encoder,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)

    date_str = datetime.now().strftime("%Y%m%d")
    if args.name is None:
        time_str = datetime.now().strftime("%H%M")
        save_dir_name = f"{date_str}--{time_str}"
    else:
        save_dir_name = f"{date_str}--{args.name}"
    save_vid_dir = os.path.join('results', save_dir_name, 'concat_vid')
    os.makedirs(save_vid_dir, exist_ok=True)
    save_split_vid_dir = os.path.join('results', save_dir_name, 'split_vid')
    os.makedirs(save_split_vid_dir, exist_ok=True)
    
    # Create temporary directory for chunks
    temp_chunks_dir = os.path.join('results', save_dir_name, 'temp_chunks')
    os.makedirs(temp_chunks_dir, exist_ok=True)

    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )

    if args.reference_image and args.driving_video:
        args.test_cases = {args.reference_image: [args.driving_video]}
    else:
        args.test_cases = OmegaConf.load(args.config)["test_cases"]

    for ref_image_path in list(args.test_cases.keys()):
        for pose_video_path in args.test_cases[ref_image_path]:
            video_name = os.path.basename(pose_video_path).split(".")[0]
            source_name = os.path.basename(ref_image_path).split(".")[0]

            vid_name = f"{source_name}_{video_name}.mp4"
            save_vid_path = os.path.join(save_vid_dir, vid_name)
            print(f"\nProcessing: {save_vid_path}")
            
            if os.path.exists(save_vid_path):
                print(f"Output already exists, skipping...")
                continue

            if ref_image_path.endswith('.mp4'):
                src_vid = VideoReader(ref_image_path)
                ref_img = src_vid[0].asnumpy()
                ref_img = Image.fromarray(ref_img).convert("RGB")
            else:
                ref_img = Image.open(ref_image_path).convert("RGB")

            # Load full video
            control = VideoReader(pose_video_path)
            total_frames = len(control)
            chunk_size = args.chunk_size
            
            # Make chunk_size divisible by 4
            chunk_size = (chunk_size // 4) * 4
            if chunk_size == 0:
                chunk_size = 4
            
            print(f"Total frames: {total_frames}")
            print(f"Chunk size: {chunk_size} frames")
            print(f"Number of chunks: {(total_frames + chunk_size - 1) // chunk_size}")

            # Prepare reference face (only once)
            ref_image_pil = ref_img.copy()
            ref_patch = crop_face(ref_image_pil, face_mesh)
            ref_face_pil = Image.fromarray(ref_patch).convert("RGB")

            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed)

            # Process video in chunks
            chunk_video_paths = []
            
            for chunk_idx in range(0, total_frames, chunk_size):
                chunk_end = min(chunk_idx + chunk_size, total_frames)
                actual_chunk_size = chunk_end - chunk_idx
                
                # Make sure chunk size is divisible by 4
                actual_chunk_size = (actual_chunk_size // 4) * 4
                if actual_chunk_size == 0:
                    continue
                
                chunk_end = chunk_idx + actual_chunk_size
                
                print(f"\n--- Processing chunk {chunk_idx // chunk_size + 1}: frames {chunk_idx} to {chunk_end-1} ---")
                
                # Load chunk frames
                sel_idx = range(chunk_idx, chunk_end)
                control_chunk = control.get_batch([list(sel_idx)]).asnumpy()  # N, H, W, C

                dri_faces = []
                ori_pose_images = []
                
                for idx_control, pose_image_pil in tqdm(enumerate(control_chunk), total=len(control_chunk), desc=f'Cropping faces (chunk {chunk_idx // chunk_size + 1})'):
                    pose_image_pil = Image.fromarray(pose_image_pil).convert("RGB")
                    ori_pose_images.append(pose_image_pil)
                    dri_face = crop_face(pose_image_pil, face_mesh)
                    dri_face_pil = Image.fromarray(dri_face).convert("RGB")
                    dri_faces.append(dri_face_pil)

                face_tensor_list = []
                ori_pose_tensor_list = []
                ref_tensor_list = []

                for idx, pose_image_pil in enumerate(ori_pose_images):
                    face_tensor_list.append(pose_transform(dri_faces[idx]))
                    ori_pose_tensor_list.append(pose_transform(pose_image_pil))
                    ref_tensor_list.append(pose_transform(ref_image_pil))

                ref_tensor = torch.stack(ref_tensor_list, dim=0)  # (f, c, h, w)
                ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)  # (c, f, h, w)

                face_tensor = torch.stack(face_tensor_list, dim=0)  # (f, c, h, w)
                face_tensor = face_tensor.transpose(0, 1).unsqueeze(0)

                ori_pose_tensor = torch.stack(ori_pose_tensor_list, dim=0)  # (f, c, h, w)
                ori_pose_tensor = ori_pose_tensor.transpose(0, 1).unsqueeze(0)

                print(f"Generating video for chunk {chunk_idx // chunk_size + 1}...")
                gen_video = pipe(
                    ori_pose_images,
                    ref_image_pil,
                    dri_faces,
                    ref_face_pil,
                    width,
                    height,
                    len(dri_faces),
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    generator=generator,
                    temporal_window_size=4,
                    temporal_adaptive_step=4,
                ).videos

                # Save chunk
                chunk_path = os.path.join(temp_chunks_dir, f'chunk_{chunk_idx:06d}.mp4')
                save_videos_grid(gen_video, chunk_path, n_rows=1, fps=25, crf=18)
                chunk_video_paths.append(chunk_path)
                
                print(f"Chunk {chunk_idx // chunk_size + 1} saved: {chunk_path}")
                
                # Clear GPU memory
                del gen_video, face_tensor, ori_pose_tensor, ref_tensor
                del face_tensor_list, ori_pose_tensor_list, ref_tensor_list
                del dri_faces, ori_pose_images
                torch.cuda.empty_cache()

            # Concatenate all chunks
            print(f"\n--- Concatenating {len(chunk_video_paths)} chunks ---")
            final_output_path = os.path.join(save_split_vid_dir, vid_name)
            
            success = concatenate_videos(chunk_video_paths, final_output_path, audio_source=pose_video_path)
            
            if success:
                print(f"✅ Final video saved: {final_output_path}")
                
                # Create concatenated version with reference/pose/output
                # For this, we'd need to regenerate or we can skip it for chunked processing
                # Since it would require loading entire video in memory
                print("Note: Concatenated grid view (4-row) skipped for chunked processing to save memory")
                
                # Clean up temporary chunks
                print("Cleaning up temporary chunks...")
                for chunk_path in chunk_video_paths:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                
                if os.path.exists(temp_chunks_dir):
                    try:
                        os.rmdir(temp_chunks_dir)
                    except:
                        pass  # Directory might not be empty
            else:
                print(f"❌ Failed to concatenate videos")

if __name__ == "__main__":
    args = parse_args()
    main(args)
