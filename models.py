"""
models.py - Convolutional Generator and Critic for WGAN-GP.

Supports:
  - 32x32  (CIFAR-10)              — 4 upsampling stages
  - 64x64  (Oxford Flowers / other) — 5 upsampling stages
  - 96x96  (STL-10 native)          — 5 upsampling (3→6→12→24→48→96) + 2 refinement = 8 layers
  - 128x128                         — 6 upsampling + 2 refinement = 8 layers

Architecture:
- Generator: Transposed Convolutional Network with Batch Normalization and Tanh output.
- Critic:    Convolutional Network with Instance Normalization (no BatchNorm in critic,
             as recommended for WGAN-GP).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper blocks
# ---------------------------------------------------------------------------

def _make_gen_block(in_ch, out_ch, kernel=4, stride=2, padding=1, final=False):
    """ConvTranspose block: BN + ReLU (or Tanh at final layer)."""
    layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False)]
    if final:
        layers.append(nn.Tanh())
    else:
        layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    return layers


def _make_critic_block(in_ch, out_ch, kernel=4, stride=2, padding=1, first=False):
    """Conv block: (optional) InstanceNorm + LeakyReLU. No norm on first layer."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)]
    if not first:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    Latent vector (z_dim,) → RGB image (3, img_size, img_size).

    img_size=32  → 4 upsampling ConvTranspose stages
    img_size=64  → 5 upsampling ConvTranspose stages
    img_size=96  → 5 upsampling (3→6→12→24→48→96) + 2 refinement Conv = 8 layers
    img_size=128 → 6 upsampling ConvTranspose + 2 refinement Conv = 8 layers
    """

    def __init__(self, z_dim=100, img_size=64, base_ch=64):
        super().__init__()
        assert img_size in (32, 64, 96, 128), "img_size must be 32, 64, 96, or 128"
        self.img_size = img_size

        if img_size == 96:
            # 8-layer architecture for STL-10 native resolution
            # Spatial: 3→6→12→24→48→96 (5 stride-2 upsamples, start at 3×3)
            ch = base_ch
            layers = [
                # Layer 1: z → 3×3  (kernel=3, stride=1, padding=0)
                *_make_gen_block(z_dim,   ch * 8, kernel=3, stride=1, padding=0),
                # Layer 2: 3 → 6
                *_make_gen_block(ch * 8,  ch * 4),
                # Layer 3: 6 → 12
                *_make_gen_block(ch * 4,  ch * 4),
                # Layer 4: 12 → 24
                *_make_gen_block(ch * 4,  ch * 2),
                # Layer 5: 24 → 48
                *_make_gen_block(ch * 2,  ch * 2),
                # Layer 6: 48 → 96
                *_make_gen_block(ch * 2,  ch),
                # Layer 7: refinement at 96×96
                nn.Conv2d(ch, ch // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(inplace=True),
                # Layer 8: final RGB output
                nn.Conv2d(ch // 2, 3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
            ]
        elif img_size == 128:
            # 8-layer architecture: 6 upsampling + 2 refinement Conv layers
            ch = base_ch
            layers = [
                # Layer 1: z → 4×4
                *_make_gen_block(z_dim,   ch * 8, kernel=4, stride=1, padding=0),
                # Layer 2: 4 → 8
                *_make_gen_block(ch * 8,  ch * 4),
                # Layer 3: 8 → 16
                *_make_gen_block(ch * 4,  ch * 4),
                # Layer 4: 16 → 32
                *_make_gen_block(ch * 4,  ch * 2),
                # Layer 5: 32 → 64
                *_make_gen_block(ch * 2,  ch * 2),
                # Layer 6: 64 → 128
                *_make_gen_block(ch * 2,  ch),
                # Layer 7: refinement at 128×128
                nn.Conv2d(ch, ch // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(inplace=True),
                # Layer 8: final RGB output
                nn.Conv2d(ch // 2, 3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
            ]
        elif img_size == 64:
            layers = [
                # z → 4×4
                *_make_gen_block(z_dim,      base_ch * 8, kernel=4, stride=1, padding=0),
                # 4 → 8
                *_make_gen_block(base_ch * 8, base_ch * 4),
                # 8 → 16
                *_make_gen_block(base_ch * 4, base_ch * 2),
                # 16 → 32
                *_make_gen_block(base_ch * 2, base_ch),
                # 32 → 64
                *_make_gen_block(base_ch,     3, final=True),
            ]
        else:  # 32
            layers = [
                # z → 4×4
                *_make_gen_block(z_dim,      base_ch * 4, kernel=4, stride=1, padding=0),
                # 4 → 8
                *_make_gen_block(base_ch * 4, base_ch * 2),
                # 8 → 16
                *_make_gen_block(base_ch * 2, base_ch),
                # 16 → 32
                *_make_gen_block(base_ch,     3, final=True),
            ]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # z shape: (N, z_dim) → (N, z_dim, 1, 1)
        return self.net(z.view(z.size(0), -1, 1, 1))


# ---------------------------------------------------------------------------
# Critic  (not a Discriminator — outputs real values, no sigmoid)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    RGB image (3, img_size, img_size) → scalar score (Wasserstein critic).

    img_size=32  → 4 downsampling stages
    img_size=64  → 5 downsampling stages
    img_size=96  → 2 refinement Conv + 5 downsampling (96→48→24→12→6→3) + final = 8 layers
    img_size=128 → 2 refinement Conv + 6 downsampling = 8 layers
    """

    def __init__(self, img_size=64, base_ch=64):
        super().__init__()
        assert img_size in (32, 64, 96, 128), "img_size must be 32, 64, 96, or 128"

        if img_size == 96:
            ch = base_ch
            layers = [
                # Layer 1: refinement at 96×96 (no downsampling, no norm on first)
                nn.Conv2d(3, ch // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # Layer 2: refinement
                nn.Conv2d(ch // 2, ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                # Layer 3: 96 → 48
                *_make_critic_block(ch,      ch * 2),
                # Layer 4: 48 → 24
                *_make_critic_block(ch * 2,  ch * 4),
                # Layer 5: 24 → 12
                *_make_critic_block(ch * 4,  ch * 4),
                # Layer 6: 12 → 6
                *_make_critic_block(ch * 4,  ch * 8),
                # Layer 7: 6 → 3
                *_make_critic_block(ch * 8,  ch * 8),
                # Layer 8: 3×3 → 1×1
                nn.Conv2d(ch * 8, 1, kernel_size=3, stride=1, padding=0, bias=False),
            ]
        elif img_size == 128:
            ch = base_ch
            layers = [
                # Layer 1: refinement at 128×128 (no downsampling, no norm on first)
                nn.Conv2d(3, ch // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # Layer 2: refinement
                nn.Conv2d(ch // 2, ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                # Layer 3: 128 → 64
                *_make_critic_block(ch,      ch * 2),
                # Layer 4: 64 → 32
                *_make_critic_block(ch * 2,  ch * 2),
                # Layer 5: 32 → 16
                *_make_critic_block(ch * 2,  ch * 4),
                # Layer 6: 16 → 8
                *_make_critic_block(ch * 4,  ch * 4),
                # Layer 7: 8 → 4
                *_make_critic_block(ch * 4,  ch * 8),
                # Layer 8: 4×4 → 1×1
                nn.Conv2d(ch * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            ]
        elif img_size == 64:
            layers = [
                # 64 → 32
                *_make_critic_block(3,          base_ch,     first=True),
                # 32 → 16
                *_make_critic_block(base_ch,     base_ch * 2),
                # 16 → 8
                *_make_critic_block(base_ch * 2, base_ch * 4),
                # 8 → 4
                *_make_critic_block(base_ch * 4, base_ch * 8),
                # 4×4 → 1×1
                nn.Conv2d(base_ch * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            ]
        else:  # 32
            layers = [
                # 32 → 16
                *_make_critic_block(3,          base_ch,     first=True),
                # 16 → 8
                *_make_critic_block(base_ch,     base_ch * 2),
                # 8 → 4
                *_make_critic_block(base_ch * 2, base_ch * 4),
                # 4×4 → 1×1
                nn.Conv2d(base_ch * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            ]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        return self.net(x).view(x.size(0))
