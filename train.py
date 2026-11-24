"""
train.py - WGAN-GP (Wasserstein GAN with Gradient Penalty) training script.

Supports:
  - CIFAR-10 (32x32), optionally filtering to cars only (class 1)
  - Oxford 102 Flowers (64x64, downloaded via torchvision)

Key features for reliable convergence:
  1. Earth Mover's (Wasserstein-1) Distance loss.
  2. Gradient Penalty (Gulrajani et al., 2017) instead of weight clipping.
  3. 5 critic updates per 1 generator update.
  4. Adam with β1=0, β2=0.9 (standard for WGAN-GP).
  5. Milestone image grids saved every --milestone_interval epochs.
"""

import os
import argparse
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import make_grid, save_image

from models import Generator, Critic


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Gradient Penalty
# ---------------------------------------------------------------------------

def compute_gradient_penalty(critic, real, fake, device):
    """
    WGAN-GP gradient penalty: enforces ||∇D(x̂)||₂ = 1 along interpolated lines.
    """
    N = real.size(0)
    alpha = torch.rand(N, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    scores = critic(interpolated)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad = grad.view(N, -1)
    penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class FlatImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted(
            os.path.join(folder, filename)
            for filename in os.listdir(folder)
            if filename.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            image = image.convert("RGB")
        return self.transform(image) if self.transform else image, 0


def get_cifar10_loader(batch_size, data_root="./data", cars_only=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)

    if cars_only:
        # CIFAR-10 class 1 = automobile
        indices = [i for i, (_, label) in enumerate(dataset) if label == 1]
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Cars-only subset: {len(dataset)} images")

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )


def get_flowers_loader(batch_size, img_size=64, data_root="./data"):
    """Oxford 102 Flowers via torchvision (requires internet on first run)."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # 'train' split has 1020 images; 'train'+'val' together = 2040
    train_set = datasets.Flowers102(root=data_root, split="train", download=True, transform=transform)
    val_set   = datasets.Flowers102(root=data_root, split="val",   download=True, transform=transform)
    full_set  = torch.utils.data.ConcatDataset([train_set, val_set])

    return torch.utils.data.DataLoader(
        full_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )


def get_stl10_loader(batch_size, img_size=96, data_root="./data"):
    """
    STL-10 unlabeled images — expects PNG files extracted by the mttk/STL10
    script at: <data_root>/STL10/unlabeled_images/*.png
    """
    img_dir = os.path.join(data_root, "STL10", "unlabeled_images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"STL-10 images not found at {img_dir}.\n"
            "Extract the dataset so that unlabeled PNGs are at data/STL10/unlabeled_images/"
        )
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = FlatImageDataset(img_dir, transform=transform)
    if not len(dataset):
        raise FileNotFoundError(
            f"No PNG or JPEG images found in {img_dir}. "
            "Extract the STL-10 unlabeled images into this directory."
        )

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def save_sample_grid(generator, fixed_z, epoch, out_dir, device, nrow=8):
    """Save a grid of generated images for the fixed noise vector."""
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_z.to(device)).cpu()
    generator.train()

    # Denormalise [-1,1] → [0,1]
    fake = (fake + 1) / 2
    grid = make_grid(fake, nrow=nrow, padding=2)
    path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    save_image(grid, path)
    return path


def save_milestone_progression(milestone_paths, out_path):
    """Create a figure showing all milestone grids side-by-side."""
    n = len(milestone_paths)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, path in zip(axes, milestone_paths):
        img = plt.imread(path)
        ax.imshow(img)
        # Extract epoch number from filename
        epoch_str = os.path.basename(path).replace("epoch_", "").replace(".png", "")
        ax.set_title(f"Epoch {int(epoch_str)}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Milestone progression saved → {out_path}")


def save_loss_plot(g_losses, c_losses, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g_losses, label="Generator Loss", alpha=0.7)
    ax.plot(c_losses, label="Critic Loss (W-dist)", alpha=0.7)
    ax.set_xlabel("Batch iteration")
    ax.set_ylabel("Loss")
    ax.set_title("WGAN-GP Training Losses")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = get_device()
    print(f"Training on: {device}")

    # --- Directories ---
    results_dir = args.results_dir
    samples_dir  = os.path.join(results_dir, "samples")
    ckpt_dir     = os.path.join(results_dir, "checkpoints")
    milestone_dir = os.path.join(results_dir, "milestones")
    for d in (samples_dir, ckpt_dir, milestone_dir):
        os.makedirs(d, exist_ok=True)

    # --- Dataset ---
    if args.stl10:
        img_size = 96
        loader = get_stl10_loader(args.batch_size, img_size=img_size, data_root=args.data_root)
        print(f"STL-10 unlabeled loader ready — {len(loader.dataset)} images, {img_size}×{img_size} (native)")
    elif args.flowers:
        img_size = 64
        loader = get_flowers_loader(args.batch_size, img_size=img_size, data_root=args.data_root)
        print(f"Oxford Flowers 102 loader ready — {len(loader.dataset)} images, {img_size}×{img_size}")
    else:
        img_size = 32
        loader = get_cifar10_loader(args.batch_size, data_root=args.data_root, cars_only=args.cars_only)
        print(f"CIFAR-10 loader ready — {len(loader.dataset)} images, 32×32")

    # --- Models ---
    netG = Generator(z_dim=args.z_dim, img_size=img_size).to(device)
    netC = Critic(img_size=img_size).to(device)
    print(f"Generator params: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Critic params:    {sum(p.numel() for p in netC.parameters()):,}")

    # --- Optimisers (β1=0 is critical for WGAN-GP) ---
    opt_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_C = optim.Adam(netC.parameters(), lr=args.lr, betas=(0.0, 0.9))

    # Fixed noise for consistent visualisation across epochs
    fixed_z = torch.randn(64, args.z_dim, device=device)

    g_losses, c_losses = [], []
    milestone_paths = []
    global_step = 0

    # --- Epoch loop ---
    for epoch in range(1, args.epochs + 1):
        netG.train()
        netC.train()
        epoch_g_loss = 0.0
        epoch_c_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                real = batch[0].to(device)
            else:
                real = batch.to(device)
            N = real.size(0)

            # ----------------------------------------------------------------
            # 1) Critic update  (n_critic times per generator step)
            # ----------------------------------------------------------------
            for _ in range(args.n_critic):
                z = torch.randn(N, args.z_dim, device=device)
                fake = netG(z).detach()

                c_real = netC(real).mean()
                c_fake = netC(fake).mean()
                gp     = compute_gradient_penalty(netC, real, fake, device)
                c_loss = c_fake - c_real + args.lambda_gp * gp

                opt_C.zero_grad()
                c_loss.backward()
                opt_C.step()

            # ----------------------------------------------------------------
            # 2) Generator update
            # ----------------------------------------------------------------
            z = torch.randn(N, args.z_dim, device=device)
            fake   = netG(z)
            g_loss = -netC(fake).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            # ---- logging ----
            g_losses.append(g_loss.item())
            c_losses.append(c_loss.item())
            epoch_g_loss += g_loss.item()
            epoch_c_loss += c_loss.item()
            n_batches    += 1
            global_step  += 1

            if batch_idx % args.log_interval == 0:
                w_dist = -(c_fake.item() - c_real.item())
                print(
                    f"Epoch [{epoch:3d}/{args.epochs}] "
                    f"Batch [{batch_idx:4d}/{len(loader)}]  "
                    f"C: {c_loss.item():+.4f}  G: {g_loss.item():+.4f}  "
                    f"W-dist: {w_dist:.4f}"
                )

        # ---- end of epoch ----
        avg_g = epoch_g_loss / max(n_batches, 1)
        avg_c = epoch_c_loss / max(n_batches, 1)
        print(f"\n=== Epoch {epoch} complete | Avg G: {avg_g:.4f} | Avg C: {avg_c:.4f} ===\n")

        # Save sample grid every epoch
        sample_path = save_sample_grid(netG, fixed_z, epoch, samples_dir, device)

        # Save milestone every milestone_interval epochs
        if epoch % args.milestone_interval == 0 or epoch == 1 or epoch == args.epochs:
            import shutil
            ms_path = os.path.join(milestone_dir, f"epoch_{epoch:04d}.png")
            shutil.copy(sample_path, ms_path)
            milestone_paths.append(ms_path)
            print(f"★ Milestone saved: {ms_path}")
            # Update progression figure after each milestone
            save_milestone_progression(
                milestone_paths,
                os.path.join(results_dir, "milestones_progression.png")
            )

        # Save checkpoint
        torch.save(netG.state_dict(), os.path.join(ckpt_dir, "generator.pth"))
        torch.save(netC.state_dict(), os.path.join(ckpt_dir, "critic.pth"))

    # Final plots
    save_loss_plot(g_losses, c_losses, os.path.join(results_dir, "training_metrics.png"))
    print("\nTraining complete!")
    print(f"  Checkpoints : {ckpt_dir}")
    print(f"  Samples     : {samples_dir}")
    print(f"  Milestones  : {os.path.join(results_dir, 'milestones_progression.png')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="WGAN-GP Training")
    p.add_argument("--epochs",             type=int,   default=50)
    p.add_argument("--batch_size",         type=int,   default=64)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--z_dim",              type=int,   default=100)
    p.add_argument("--n_critic",           type=int,   default=5,
                   help="Critic updates per generator update")
    p.add_argument("--lambda_gp",          type=float, default=10.0,
                   help="Gradient penalty coefficient")
    p.add_argument("--log_interval",       type=int,   default=50)
    p.add_argument("--milestone_interval", type=int,   default=10,
                   help="Save milestone image grid every N epochs")
    p.add_argument("--data_root",          type=str,   default="./data")
    p.add_argument("--results_dir",        type=str,   default="./results")
    p.add_argument("--cars_only",          action="store_true",
                   help="CIFAR-10: train only on automobile class")
    p.add_argument("--flowers",            action="store_true",
                   help="Use Oxford 102 Flowers dataset at 64×64")
    p.add_argument("--stl10",              action="store_true",
                   help="Use STL-10 unlabeled split at 128×128 (100k images)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
