#!/usr/bin/env python
import argparse, os, sys, random
import numpy as np
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint

def pick_device(force: str | None = None) -> torch.device:
    if force:
        return torch.device(force)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_ply(path: str, xyz: np.ndarray, rgb: np.ndarray | None):
    xyz = np.asarray(xyz, dtype=np.float32)
    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)
    else:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=False, default=os.environ.get("POINT_E_PROMPT", "a shiny red sports car"))
    ap.add_argument("--out", default=os.environ.get("POINT_E_OUT", "data/outputs/pointclouds") + "/output.ply")
    ap.add_argument("--device", default=None, help="mps|cuda|cpu (default: auto)")
    ap.add_argument("--no_upsample", action="store_true", help="use base only (1024 pts)")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[info] Python: {sys.version.split()[0]}")
    print(f"[info] CWD: {os.getcwd()}")
    print(f"[info] Args: {args}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"[info] Device: {device} | CUDA avail: {torch.cuda.is_available()} | MPS avail: {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")

    # ----- load models -----
    base_name = "base40M-textvec"
    print(f"[info] Loading base model: {base_name}")
    base_model = model_from_config(MODEL_CONFIGS[base_name], device).eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    base_model.load_state_dict(load_checkpoint(base_name, device))
    models = [base_model]
    diffs = [base_diffusion]
    num_points = [1024]  # base MUST be 1024

    if not args.no_upsample:
        print("[info] Loading upsampler…")
        upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device).eval()
        upsampler_diff = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])
        upsampler_model.load_state_dict(load_checkpoint("upsample", device))
        models.append(upsampler_model)
        diffs.append(upsampler_diff)
        num_points.append(3072)  # ~4096 total

    sampler = PointCloudSampler(
        device=device,
        models=models,
        diffusions=diffs,
        num_points=num_points,
        aux_channels=["R","G","B"],
        # text-guided base, unconditioned upsampler (common recipe)
        guidance_scale=[args.guidance] if args.no_upsample else [args.guidance, 0.0],
        model_kwargs_key_filter=("texts","")
    )
    print(f"[info] Sampler ready. Prompt: {args.prompt!r}")

    # ----- sample -----
    print("[info] Sampling… (this can take 1–3 minutes on MPS)")
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[args.prompt]))):
        samples = x
    print("[info] Sampling done ✓")

    # ----- convert and save -----
    pc = sampler.output_to_point_clouds(samples)[0]
    xyz = to_numpy(getattr(pc, "coords", pc))[:, :3]

    # handle colors as 0–1 floats or 0–255 ints robustly
    if hasattr(pc, "channels") and all(k in pc.channels for k in ("R","G","B")):
        R = to_numpy(pc.channels["R"]); G = to_numpy(pc.channels["G"]); B = to_numpy(pc.channels["B"])
        if R.max() > 1.5 or G.max() > 1.5 or B.max() > 1.5:
            rgb = np.stack([R/255.0, G/255.0, B/255.0], axis=1)
        else:
            rgb = np.stack([R, G, B], axis=1)
    else:
        rgb = np.ones((xyz.shape[0], 3), dtype=float)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    save_ply(args.out, xyz, rgb)
    print(f"[done] Saved: {args.out}  | points={xyz.shape[0]}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(1)
