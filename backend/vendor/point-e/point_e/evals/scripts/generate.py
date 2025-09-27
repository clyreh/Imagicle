#!/usr/bin/env python3.12
import os, numpy as np, torch, open3d as o3d
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def build_sampler(device):
    base_name, upsampler_name = 'base40M-textvec', 'upsample'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device); base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    base_model.load_state_dict(load_checkpoint(base_name, device))

    upsampler_model = model_from_config(MODEL_CONFIGS[upsampler_name], device); upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[upsampler_name])
    upsampler_model.load_state_dict(load_checkpoint(upsampler_name, device))

    return PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096-1024],
        aux_channels=['R','G','B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', '')
    )

def to_open3d_point_cloud(pc):
    coords = pc.coords
    r = pc.channels['R'].astype(np.float32)/255.0
    g = pc.channels['G'].astype(np.float32)/255.0
    b = pc.channels['B'].astype(np.float32)/255.0
    colors = np.stack([r,g,b],1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    prompt = os.environ.get("POINT_E_PROMPT", "a shiny red sports car")
    out_dir = os.environ.get("POINT_E_OUT", "backend/data/outputs/pointclouds")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.ply")

    device = get_device()
    print("Using device:", device)
    sampler = build_sampler(device)
    print("Prompt:", prompt)

    samples = None
    for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    pcd = to_open3d_point_cloud(pc)
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
