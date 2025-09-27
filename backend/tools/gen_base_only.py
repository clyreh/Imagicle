import os, numpy as np, torch
from plyfile import PlyData, PlyElement
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint

def to_ply_arrays(pc):
    # Convert point_e PointCloud to numpy arrays
    xyz = pc.coords.astype(np.float32)
    r = pc.channels.get('R', np.zeros(len(xyz), dtype=np.uint8))
    g = pc.channels.get('G', np.zeros(len(xyz), dtype=np.uint8))
    b = pc.channels.get('B', np.zeros(len(xyz), dtype=np.uint8))
    rgb = np.stack([r,g,b], axis=1).astype(np.uint8)
    return xyz, rgb

def write_ascii_ply(path, xyz, rgb=None):
    if rgb is None:
        rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
    verts = np.empty(xyz.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4'),
                                          ('red','u1'),('green','u1'),('blue','u1')])
    verts['x'], verts['y'], verts['z'] = xyz[:,0], xyz[:,1], xyz[:,2]
    verts['red'], verts['green'], verts['blue'] = rgb[:,0], rgb[:,1], rgb[:,2]
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=True).write(path)

def main():
    prompt = os.environ.get("POINT_E_PROMPT", "a bonsai tree in a pot")
    out_dir = os.environ.get("POINT_E_OUT", "backend/data/outputs/pointclouds")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output_base_only.ply")

    # pick GPU if available in the environment you run this in (VM container)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load base text-conditioned model (no upsampler)
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device); base_model.eval()
    base_diff = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    base_model.load_state_dict(load_checkpoint(base_name, device))

    # build sampler with only the base stage (fast, ~1k points)
    sampler = PointCloudSampler(
        device=device,
        models=[base_model],
        diffusions=[base_diff],
        num_points=[1024],
        aux_channels=['R','G','B'],
        guidance_scale=[3.0],
        model_kwargs_key_filter=('texts', '')
    )

    print("Prompt:", prompt)
    samples = None
    # iterate progressive samples; keep the last (final) one
    for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    xyz, rgb = to_ply_arrays(pc)
    write_ascii_ply(out_path, xyz, rgb)
    print("Saved:", out_path, "points:", len(xyz))

if __name__ == "__main__":
    main()
