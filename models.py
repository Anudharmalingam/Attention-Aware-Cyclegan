"""
models.py
Attention-Aware CycleGAN (A-CycleGAN) architecture for CT<->MRI translation.

Generator:
    ResNet-based encoder-decoder (as in Zhu et al. CycleGAN, ICCV 2017) with an
    added Self-Attention block (Zhang et al., SAGAN) in the bottleneck so the
    generator itself can focus on anatomically salient regions.

Discriminator:
    Attention-gated PatchGAN discriminator + variational-encoding (VAE) branch,
    following Kearney et al., "Attention-Aware Discrimination for MR-to-CT Image
    Translation Using Cycle-Consistent GANs", Radiol Artif Intell 2020;2(2):e190027
    (https://pubmed.ncbi.nlm.nih.gov/33937817/). The attention-gate module gates
    shallow feature maps (input/skip signal) using deeper feature maps (gating
    signal), and a small VAE head on the bottleneck regularises the latent space,
    allowing a deeper discriminator to be used without destabilising GAN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def weights_init_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "InstanceNorm2d" in classname or "BatchNorm2d" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class SelfAttention(nn.Module):
    """Self-attention (SAGAN-style) — lets the generator relate distant spatial
    positions, useful for consistent global anatomy (skull shape, ventricles, etc.)."""

    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key   = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)   # B, N, C'
        k = self.key(x).view(B, -1, H * W)                       # B, C', N
        attn = self.softmax(torch.bmm(q, k))                     # B, N, N
        v = self.value(x).view(B, -1, H * W)                     # B, C, N
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #

class AttentionResNetGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ngf=64, n_res_blocks=9, use_attention=True):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        c = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2

        # Bottleneck residual blocks (+ self-attention in the middle)
        mid = n_res_blocks // 2
        for _ in range(mid):
            layers.append(ResidualBlock(c))
        if use_attention:
            layers.append(SelfAttention(c))
        for _ in range(n_res_blocks - mid):
            layers.append(ResidualBlock(c))

        # Upsampling
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, out_ch, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------------------------------- #
# Attention-Gated, VAE-enhanced Discriminator  (the paper's core contribution)
# --------------------------------------------------------------------------- #

class AttentionGate(nn.Module):
    """Gating mechanism: a deeper 'gating' feature map (small spatial size,
    high semantic content) is upsampled and used to re-weight a shallower
    'skip' feature map, exactly as in Fig. 1 of Kearney et al. 2020:
        a_g = skip * sigmoid( W_g(gate_upsampled) + W_x(skip) )
    """

    def __init__(self, skip_ch, gate_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1),
            nn.InstanceNorm2d(inter_ch),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1),
            nn.InstanceNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip, gate):
        gate_up = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        combined = self.relu(self.W_gate(gate_up) + self.W_skip(skip))
        alpha = self.psi(combined)              # attention coefficients, shape B,1,H,W
        return skip * alpha, alpha


class VAEHead(nn.Module):
    """Small variational bottleneck. Predicts mu/logvar from the deepest feature
    map and reparameterises; regularises the discriminator's latent space so a
    deeper network can be used without over-fitting / mode collapse (per paper)."""

    def __init__(self, in_ch, latent_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(in_ch, latent_dim)
        self.fc_logvar = nn.Linear(in_ch, latent_dim)
        self.fc_back = nn.Linear(latent_dim, in_ch)

    def forward(self, x):
        b, c, h, w = x.shape
        pooled = self.pool(x).view(b, c)
        mu, logvar = self.fc_mu(pooled), self.fc_logvar(pooled)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        z_map = self.fc_back(z).view(b, c, 1, 1).expand(-1, -1, h, w)
        return x + z_map, mu, logvar


class AttentionVAEDiscriminator(nn.Module):
    """PatchGAN-style discriminator with attention gating between successive
    feature scales, plus an optional VAE bottleneck (USE_VAE_DISC)."""

    def __init__(self, in_ch=1, ndf=64, use_vae=True):
        super().__init__()
        self.use_vae = use_vae

        def block(i, o, norm=True, stride=2):
            layers = [nn.Conv2d(i, o, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.stage1 = block(in_ch, ndf, norm=False)     # H/2
        self.stage2 = block(ndf, ndf * 2)                # H/4
        self.stage3 = block(ndf * 2, ndf * 4)             # H/8
        self.stage4 = block(ndf * 4, ndf * 8)             # H/16

        # attention gates: stage4 (deep, semantic) gates stage3 and stage2
        self.gate_32 = AttentionGate(skip_ch=ndf * 4, gate_ch=ndf * 8, inter_ch=ndf * 4)
        self.gate_21 = AttentionGate(skip_ch=ndf * 2, gate_ch=ndf * 8, inter_ch=ndf * 2)

        if use_vae:
            self.vae = VAEHead(ndf * 8, latent_dim=128)

        # final patch-classification head, fed by concatenation of gated features
        self.head = nn.Sequential(
            nn.Conv2d(ndf * 8 + ndf * 4 + ndf * 2, ndf * 4, 3, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 3, padding=1),   # patch logits (no sigmoid; used with LSGAN/MSE loss)
        )

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        mu = logvar = None
        if self.use_vae:
            f4, mu, logvar = self.vae(f4)

        gated_f3, attn_32 = self.gate_32(f3, f4)
        gated_f2, attn_21 = self.gate_21(f2, f4)

        gated_f3_up = F.interpolate(gated_f3, size=f4.shape[2:], mode="bilinear", align_corners=False)
        gated_f2_up = F.interpolate(gated_f2, size=f4.shape[2:], mode="bilinear", align_corners=False)

        merged = torch.cat([f4, gated_f3_up, gated_f2_up], dim=1)
        patch_logits = self.head(merged)

        extras = {
            "mu": mu, "logvar": logvar,
            "attn_maps": [attn_32, attn_21],   # useful for the paper's Fig-4-style visualisations
        }
        return patch_logits, extras


# --------------------------------------------------------------------------- #
# Simple CT-vs-MRI classifier, used ONLY as an auxiliary evaluation metric
# ("fool rate" / perceptual realism accuracy — see metrics.py). It is trained
# once on REAL images and then frozen; it is NOT part of the GAN objective.
# --------------------------------------------------------------------------- #

class ModalityClassifier(nn.Module):
    def __init__(self, in_ch=1, ndf=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 4, 1),   # logit: >0 => MRI, <0 => CT (BCEWithLogits)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
