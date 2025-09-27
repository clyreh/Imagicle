#!/usr/bin/env python3
import os, numpy as np, torch
from plyfile import PlyData, PlyElement

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint

#from lora.inject import inject_lora  # optional LoRA

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def write_ascii_ply(path, coords_f32, colors_u8):
    """Write an ASCII .ply without Open3D (headless-safe)."""
    verts = np.empty(coords_f32.shape[0], dtype=[
        ('x','f4'),('y','f4'),('z','f4'),
        ('red','u1'),('green','u1'),('blue','u1')
    ])
    verts['x'], verts['y'], verts['z'] = coords_f32[:,0], coords_f32[:,1], coords_f32[:,2]
    verts['red'], verts['green'], verts['blue'] = colors_u8[:,0], colors_u8[:,1], colors_u8[:,2]
    PlyData([PlyElement.describe(verts, 'vertex')], text=True).write(path)

def build_sampler(device):
    base_name = 'base40M-textvec'

    base_model = model_from_config(MODEL_CONFIGS[base_name], device); base_model.eval()
    base_diff  = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    base_model.load_state_dict(load_checkpoint(base_name, device))

    # Single-stage sampler
    return PointCloudSampler(
        device=device,
        models=[base_model],              # 1 stage
        diffusions=[base_diff],           # 1 stage
        num_points=[4096],                # 1 entry
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0],             # 1 entry
        use_karras=[False],               # you can set True if you prefer; either way…
        karras_steps=[64],                # …must still be length-1 for this version
        model_kwargs_key_filter=('texts',),  # 1-tuple (trailing comma matters)
    )

def main():
    prompt  = os.environ.get("POINT_E_PROMPT", "a shiny red sports car")
    out_dir = os.environ.get("POINT_E_OUT", "backend/data/outputs/pointclouds")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.ply")

    device = get_device()
    print("Using device:", device)
    sampler = build_sampler(device)
    print("Prompt:", prompt)

    # Run progressive sampling (keep last sample)
    samples = None
    for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
        samples = x

    # Convert to arrays for PLY writing
    pc    = sampler.output_to_point_clouds(samples)[0]
    xyz   = pc.coords.astype(np.float32)
    r     = pc.channels.get('R', np.zeros(len(xyz), dtype=np.uint8))
    g     = pc.channels.get('G', np.zeros(len(xyz), dtype=np.uint8))
    b     = pc.channels.get('B', np.zeros(len(xyz), dtype=np.uint8))
    rgb   = np.stack([r,g,b], axis=1).astype(np.uint8)

    write_ascii_ply(out_path, xyz, rgb)
    print("Saved:", out_path, "points:", len(xyz))

if __name__ == "__main__":
    main()
