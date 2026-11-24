"""
generate.py - Generate images using a trained WGAN-GP generator.

Usage:
    # Sample 64 images
    python generate.py --checkpoint results/checkpoints/generator.pth --num_images 64

    # Latent space interpolation between two random points
    python generate.py --checkpoint results/checkpoints/generator.pth --interpolate --steps 10

    # For flowers model (64x64)
    python generate.py --checkpoint results_flowers/checkpoints/generator.pth --img_size 64
"""

import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from models import Generator


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_generator(checkpoint_path, z_dim=100, img_size=64, device="cpu"):
    netG = Generator(z_dim=z_dim, img_size=img_size).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(state)
    netG.eval()
    print(f"Generator loaded from: {checkpoint_path}")
    return netG


def generate_samples(netG, num_images, z_dim, device, nrow=8):
    z = torch.randn(num_images, z_dim, device=device)
    with torch.no_grad():
        fake = netG(z).cpu()
    fake = (fake + 1) / 2  # [-1,1] → [0,1]
    return make_grid(fake, nrow=nrow, padding=2, normalize=False)


def interpolate_latents(netG, steps, z_dim, device):
    """Spherical interpolation between two random latent vectors."""
    z1 = torch.randn(1, z_dim, device=device)
    z2 = torch.randn(1, z_dim, device=device)

    # Normalise to unit sphere
    z1 = z1 / z1.norm()
    z2 = z2 / z2.norm()

    frames = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        # Spherical linear interpolation (slerp)
        omega = torch.acos((z1 * z2).sum().clamp(-1, 1))
        if omega.abs() < 1e-6:
            z_interp = (1 - t) * z1 + t * z2
        else:
            z_interp = (torch.sin((1 - t) * omega) * z1 + torch.sin(t * omega) * z2) / torch.sin(omega)
        with torch.no_grad():
            img = netG(z_interp).cpu()
        frames.append((img + 1) / 2)

    return make_grid(torch.cat(frames, dim=0), nrow=steps, padding=2, normalize=False)


def main():
    p = argparse.ArgumentParser(description="WGAN-GP Image Generator")
    p.add_argument("--checkpoint",   type=str,   default="results/checkpoints/generator.pth")
    p.add_argument("--output",       type=str,   default="generated.png")
    p.add_argument("--num_images",   type=int,   default=64)
    p.add_argument("--z_dim",        type=int,   default=100)
    p.add_argument("--img_size",     type=int,   default=64, choices=[32, 64],
                   help="Must match the checkpoint's training resolution")
    p.add_argument("--interpolate",  action="store_true",
                   help="Generate latent-space interpolation instead of samples")
    p.add_argument("--steps",        type=int,   default=10,
                   help="Number of interpolation steps (used with --interpolate)")
    p.add_argument("--nrow",         type=int,   default=8)
    args = p.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Train first:  python train.py --flowers --epochs 50")
        return

    netG = load_generator(args.checkpoint, z_dim=args.z_dim, img_size=args.img_size, device=device)

    if args.interpolate:
        print(f"Generating {args.steps}-step latent interpolation...")
        grid = interpolate_latents(netG, args.steps, args.z_dim, device)
    else:
        print(f"Generating {args.num_images} sample images...")
        grid = generate_samples(netG, args.num_images, args.z_dim, device, nrow=args.nrow)

    save_image(grid, args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
