# Copyright 2025 Adobe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
3D Face Reconstruction Inference Pipeline

This module provides a complete pipeline for generating 3D face reconstructions
from single input images using multi-view diffusion and Gaussian splatting.
"""

import gc
import os
import yaml
import json
import importlib
import warnings
from typing import List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from einops import rearrange
from easydict import EasyDict as edict
from rich import print
from rembg import remove
from utils_folder.face_utils import REMBG_SESSION
from facenet_pytorch import MTCNN
from huggingface_hub import snapshot_download

from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from utils_folder.face_utils import preprocess_image, preprocess_image_without_cropping

# Suppress FutureWarning from facenet_pytorch
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

# Configuration constants
DEFAULT_IMG_SIZE = 512
HF_REPO_ID = "wlyu/OpenFaceLift"

def download_weights_from_hf() -> str:
    """Download model weights from HuggingFace if not already present.
    
    Returns:
        Path to the downloaded repository
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Check if weights already exist locally
    mvdiffusion_path = os.path.join(script_directory, "checkpoints/mvdiffusion/pipeckpts")
    gslrm_path = os.path.join(script_directory, "checkpoints/gslrm/ckpt_0000000000021125.pt")
    prompt_embeds_path = os.path.join(script_directory, "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt")
    
    if os.path.exists(mvdiffusion_path) and os.path.exists(gslrm_path) and os.path.exists(prompt_embeds_path):
        print("Using local model weights")
        return script_directory
    
    print(f"Downloading model weights from HuggingFace: {HF_REPO_ID}")
    print("This may take a few minutes on first run...")
    
    # Download to local directory
    cache_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=os.path.join(script_directory, "checkpoints"),
        local_dir_use_symlinks=False,
    )
    
    print("Model weights downloaded successfully!")
    return script_directory

def get_model_paths() -> Tuple[str, str, str]:
    """Get paths to model checkpoints and config files."""
    script_directory = download_weights_from_hf()
    mvdiffusion_checkpoint_path = os.path.join(script_directory, "checkpoints/mvdiffusion/pipeckpts")
    gslrm_checkpoint_path = os.path.join(script_directory, "checkpoints/gslrm/ckpt_0000000000021125.pt")
    gslrm_config_path = os.path.join(script_directory, "configs/gslrm.yaml")
    return mvdiffusion_checkpoint_path, gslrm_checkpoint_path, gslrm_config_path



def initialize_face_detector(device: torch.device) -> MTCNN:
    """Initialize face detector."""
    return MTCNN(
        image_size=512, 
        margin=0, 
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], 
        factor=0.709,
        post_process=True, 
        device=device
    )


def initialize_mvdiffusion_pipeline(mvdiffusion_checkpoint_path: str, device: torch.device):
    """Initialize MV Diffusion pipeline."""
    script_directory = download_weights_from_hf()
    
    diffusion_pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        mvdiffusion_checkpoint_path,
        torch_dtype=torch.float16,
    )
    diffusion_pipeline.unet.enable_xformers_memory_efficient_attention()
    diffusion_pipeline.to(device)
    random_generator = torch.Generator(device=diffusion_pipeline.unet.device)
    
    color_prompt_embeddings = torch.load(
        os.path.join(script_directory, "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt")
    )
    
    return diffusion_pipeline, random_generator, color_prompt_embeddings


def initialize_gslrm_model(gslrm_checkpoint_path: str, gslrm_config_path: str, device: torch.device):
    """Initialize GSLRM model."""
    model_config = edict(yaml.safe_load(open(gslrm_config_path, "r")))
    module_name, class_name = model_config.model.class_name.rsplit(".", 1)
    print(f"Loading model from {module_name} -> {class_name}")
    
    ModelClass = importlib.import_module(module_name).__dict__[class_name]
    gslrm_model = ModelClass(model_config)
    model_checkpoint = torch.load(gslrm_checkpoint_path, map_location="cpu")
    gslrm_model.load_state_dict(model_checkpoint["model"])
    gslrm_model = gslrm_model.to(device)
    gslrm_model.eval()

    return gslrm_model


def setup_camera_parameters(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Setup camera parameters for 6 views using local opencv_cameras.json."""
    script_directory = download_weights_from_hf()
    camera_file = os.path.join(script_directory, "utils_folder/opencv_cameras.json")
    
    with open(camera_file, 'r') as f:
        camera_data = json.load(f)["frames"]
    
    # Always use 6 views with indices [2, 1, 0, 5, 4, 3] like in gradio_app.py
    camera_indices = [2, 1, 0, 5, 4, 3]
    selected_cameras = [camera_data[i] for i in camera_indices]

    camera_intrinsics_list, camera_extrinsics_list = [], []
    for camera_frame in selected_cameras:
        camera_intrinsics_list.append(np.array([camera_frame["fx"], camera_frame["fy"], camera_frame["cx"], camera_frame["cy"]]))
        camera_extrinsics_list.append(np.linalg.inv(np.array(camera_frame["w2c"])))
    
    camera_intrinsics_array = np.stack(camera_intrinsics_list, axis=0).astype(np.float32)
    camera_extrinsics_array = np.stack(camera_extrinsics_list, axis=0).astype(np.float32)

    camera_intrinsics_tensor = torch.from_numpy(camera_intrinsics_array).float()[None].to(device)
    camera_extrinsics_tensor = torch.from_numpy(camera_extrinsics_array).float()[None].to(device)
    
    return camera_intrinsics_tensor, camera_extrinsics_tensor


def process_single_image(
    image_file: str,
    input_dir: str,
    output_dir: str,
    auto_crop: bool,
    unclip_pipeline,
    generator: torch.Generator,
    color_prompt_embedding: torch.Tensor,
    gs_lrm_model,
    demo_fxfycxcy: torch.Tensor,
    demo_c2w: torch.Tensor,
    guidance_scale_2D: float,
    step_2D: int,
    face_detector: Optional[MTCNN] = None
) -> None:
    """Process a single image through the 3D reconstruction pipeline."""
    print("sped up version")
    import time
    pipeline_start = time.time()
    print(f"Processing {image_file}")
    image_name = image_file.split(".")[0]

    input_image = Image.open(os.path.join(input_dir, image_file))
    input_image_np = np.array(input_image)

    demo_output_local_dir = os.path.join(output_dir, image_name)
    os.makedirs(demo_output_local_dir, exist_ok=True)

    # Preprocess image
    t0 = time.time()
    try:
        if auto_crop:
            input_image = preprocess_image(input_image_np)
        else:
            input_image = preprocess_image_without_cropping(input_image_np)
    except Exception as e:
        print(f"Failed to process {image_file}: {e}, applying fallback processing")
        try:
            input_image = remove(input_image, session=REMBG_SESSION)
            input_image = input_image.resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), Image.LANCZOS)
        except Exception as e2:
            print(f"Background removal also failed: {e2}, using original image")
            input_image = input_image.resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), Image.LANCZOS)
    print(f"[TIMING] Preprocessing: {time.time() - t0:.2f}s")

    input_image.save(os.path.join(demo_output_local_dir, "input.png"))

    # Generate multi-view images
    t0 = time.time()
    mv_imgs = unclip_pipeline(
        input_image,
        None,
        prompt_embeds=color_prompt_embedding,
        guidance_scale=guidance_scale_2D,
        num_images_per_prompt=1,
        num_inference_steps=step_2D,
        generator=generator,
        eta=1.0,
    ).images
    print(f"[TIMING] Multi-view diffusion ({step_2D} steps): {time.time() - t0:.2f}s")

    # Always use 6 views
    if len(mv_imgs) == 7:
        views = [mv_imgs[i] for i in [1, 2, 3, 4, 5, 6]]
    elif len(mv_imgs) == 6:
        views = [mv_imgs[i] for i in [0, 1, 2, 3, 4, 5]]
    else:
        raise ValueError(f"Unexpected number of views: {len(mv_imgs)}")

    # Save multi-view image
    lrm_input_save = Image.new("RGB", (DEFAULT_IMG_SIZE * len(mv_imgs), DEFAULT_IMG_SIZE))
    for i, view in enumerate(mv_imgs):
        lrm_input_save.paste(view, (DEFAULT_IMG_SIZE * i, 0))
    lrm_input_save.save(os.path.join(demo_output_local_dir, "multiview.png"))

    # Prepare input for 3D reconstruction
    lrm_input = np.stack([np.array(view) for view in views], axis=0)
    lrm_input = torch.from_numpy(lrm_input).float()[None].to(demo_fxfycxcy.device) / 255
    lrm_input = rearrange(lrm_input, "b v h w c -> b v c h w")

    index = torch.stack([
        torch.zeros(lrm_input.size(1)).long(),
        torch.arange(lrm_input.size(1)).long(),
    ], dim=-1)
    demo_index = index[None].to(demo_fxfycxcy.device)

    # Create batch
    batch = edict({
        "image": lrm_input,
        "c2w": demo_c2w,
        "fxfycxcy": demo_fxfycxcy,
        "index": demo_index,
    })

    # 3D reconstruction inference.
    # split_data=False skips the model's internal rendering pass, which would otherwise
    # rasterize all 1.57M pixel-aligned Gaussians and allocate a BinningState sort buffer
    # of ~130+ TiB — far beyond available GPU memory.
    t0 = time.time()
    with torch.no_grad():
        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
            result = gs_lrm_model.forward(batch, create_visual=False, split_data=False)
    print(f"[TIMING] GSLRM reconstruction: {time.time() - t0:.2f}s")

    # Save Gaussian splatting result
    _n_before_filter = result.gaussians[0]._xyz.shape[0]
    print(f"[DEBUG] Gaussians BEFORE apply_all_filters: {_n_before_filter:,}")

    t0 = time.time()
    filtered_gaussians = result.gaussians[0].apply_all_filters(
        opacity_thres=0.04,
        scaling_thres=0.1,
        floater_thres=0.6,
        crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
        cam_origins=None,
        nearfar_percent=(0.0001, 1.0),
    )
    _n_after_filter = result.gaussians[0]._xyz.shape[0]
    print(f"[DEBUG] Gaussians AFTER apply_all_filters: {_n_after_filter:,} "
          f"(removed {_n_before_filter - _n_after_filter:,})")
    print(f"[TIMING] apply_all_filters: {time.time() - t0:.2f}s")

    t0 = time.time()
    filtered_gaussians.save_ply(os.path.join(demo_output_local_dir, "gaussians.ply"))
    print(f"[TIMING] save_ply: {time.time() - t0:.2f}s")

    print(f"[TIMING] Total: {time.time() - pipeline_start:.2f}s")

    # Release GPU memory before processing the next image
    del result
    gc.collect()
    torch.cuda.empty_cache()


def process_images(
    input_dir: str,
    output_dir: str,
    auto_crop: bool,
    unclip_pipeline,
    generator: torch.Generator,
    color_prompt_embedding: torch.Tensor,
    gs_lrm_model,
    demo_fxfycxcy: torch.Tensor,
    demo_c2w: torch.Tensor,
    guidance_scale_2D: float,
    step_2D: int,
    face_detector: Optional[MTCNN] = None
) -> None:
    """Process all images in the input directory."""
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
        
    image_files = sorted(os.listdir(input_dir))
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    for image_file in image_files:
        if not image_file.lower().endswith(valid_extensions):
            continue
            
        process_single_image(
            image_file, input_dir, output_dir, auto_crop,
            unclip_pipeline, generator, color_prompt_embedding,
            gs_lrm_model, demo_fxfycxcy, demo_c2w,
            guidance_scale_2D, step_2D, face_detector
        )


def main(
    input_dir: str = None,
    output_dir: str = None,
    auto_crop: bool = True,
    seed: int = 4,
    guidance_scale_2D: float = 3.0,
    step_2D: int = 50
) -> None:
    """Main function for 3D face reconstruction inference.
    
    Args:
        input_dir: Input directory containing images (default: examples/)
        output_dir: Output directory for results (default: outputs/)
        auto_crop: Auto crop the face (default: True)
        seed: Random seed for generating multi-view images (default: 4)
        guidance_scale_2D: Guidance scale for generating multi-view images (default: 3.0)
        step_2D: Number of steps for generating multi-view images (default: 50)
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Set default paths if not provided
    if input_dir is None:
        input_dir = os.path.join(script_directory, "examples")
    if output_dir is None:
        output_dir = os.path.join(script_directory, "outputs")
    
    # Setup device and paths
    computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mvdiffusion_checkpoint_path, gslrm_checkpoint_path, gslrm_config_path = get_model_paths()
    
    os.makedirs(output_dir, exist_ok=True)

    face_detector = None
    if auto_crop:
        face_detector = initialize_face_detector(computation_device)

    # Initialize models
    diffusion_pipeline, random_generator, color_prompt_embeddings = initialize_mvdiffusion_pipeline(
        mvdiffusion_checkpoint_path, computation_device
    )
    gslrm_model = initialize_gslrm_model(gslrm_checkpoint_path, gslrm_config_path, computation_device)

    # Setup camera parameters (always 6 views)
    camera_intrinsics_tensor, camera_extrinsics_tensor = setup_camera_parameters(computation_device)
    
    # Set random seed
    random_generator.manual_seed(seed)

    # Process images
    process_images(
        input_dir, 
        output_dir, 
        auto_crop,
        diffusion_pipeline,
        random_generator,
        color_prompt_embeddings,
        gslrm_model,
        camera_intrinsics_tensor,
        camera_extrinsics_tensor,
        guidance_scale_2D,
        step_2D,
        face_detector
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="3D Face Reconstruction Inference Pipeline")
    parser.add_argument("--input_dir", "-i", type=str, help="Input directory containing images")
    parser.add_argument("--output_dir", "-o", type=str, help="Output directory for results")
    parser.add_argument("--auto_crop", action="store_true", default=True, help="Auto crop the face")
    parser.add_argument("--seed", type=int, default=4, help="Random seed")
    parser.add_argument("--guidance_scale_2D", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--step_2D", type=int, default=50, help="Number of diffusion steps")
    
    args = parser.parse_args()
    
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        auto_crop=args.auto_crop,
        seed=args.seed,
        guidance_scale_2D=args.guidance_scale_2D,
        step_2D=args.step_2D
    )
