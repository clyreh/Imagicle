#!/usr/bin/env python3
import os, sys, numpy as np, torch
from PIL import Image
import open3d as o3d

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_image_grid(path, size=224):
    """Return a list (grid) of PIL images; for 1 image it's a 1x1 grid."""
    img = Image.open(path).convert("RGB").resize((size, size))
    return [img]  # sampler expects list-of-images (grid)

def to_open3d_point_cloud(pc):
    coords = pc.coords
    r = pc.channels['R'].astype(np.float32)/255.0
    g = pc.channels['G'].astype(np.float32)/255.0
    b = pc.channels['B'].astype(np.float32)/255.0
    colors = np.stack([r,g,b], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    img_path = os.environ.get("POINT_E_IMAGE", "").strip()
    out_dir  = os.environ.get("POINT_E_OUT", "backend/data/outputs/pointclouds")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.ply")

    if not img_path or not os.path.exists(img_path):
        print(f"[ERR] set POINT_E_IMAGE to an existing image (got '{img_path}')", file=sys.stderr)
        sys.exit(2)

    device = get_device()
    print("Using device:", device)
    print("Image:", img_path)

    # Image-conditioned base (expects 'images' in model_kwargs) + upsampler
    base_name      = "base300M"
    upsampler_name = "upsample"

    base_model = model_from_config(MODEL_CONFIGS[base_name], device); base_model.eval()
    base_diff  = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    base_model.load_state_dict(load_checkpoint(base_name, device))

    # Optional: apply LoRA adapter for image-cond base
    lora_path = os.environ.get("POINT_E_LORA_IMAGE", "").strip()
    if lora_path:
        from lora.inject import inject_lora
        base_model = inject_lora(base_model, r=16, alpha=16, dropout=0.0, verbose=False)
        sd = torch.load(lora_path, map_location="cpu")
        base_model.load_state_dict(sd, strict=False)
        print(f"[LoRA] loaded image-cond adapter: {lora_path}")

    up_model = model_from_config(MODEL_CONFIGS[upsampler_name], device); up_model.eval()
    up_diff  = diffusion_from_config(DIFFUSION_CONFIGS[upsampler_name])
    up_model.load_state_dict(load_checkpoint(upsampler_name, device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, up_model],
        diffusions=[base_diff, up_diff],
        num_points=[1024, 4096-1024],
        aux_channels=['R','G','B'],
        guidance_scale=[0.0, 0.0],      # image-cond usually with little/no CFG
        use_karras=[True, True],
        model_kwargs_key_filter=('images', '')
    )

    images = load_image_grid(img_path)
    last = None
    for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[images])):
        last = x

    pc = sampler.output_to_point_clouds(last)[0]
    pcd = to_open3d_point_cloud(pc)
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
    print("Saved:", out_path, "points:", len(pc.coords))

if __name__ == "__main__":
    main()
