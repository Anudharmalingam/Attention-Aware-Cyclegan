"""
metrics.py
Evaluation metrics for CT<->MRI image translation, chosen to match what
reviewers expect for this exact task (these are the metrics used by Kearney
et al. 2020 and by essentially every CT/MRI CycleGAN paper since):

    - MAE   : mean absolute error in de-normalised intensity units
    - PSNR  : peak signal-to-noise ratio (dB)
    - SSIM  : structural similarity index
    - FID   : Frechet Inception Distance between real-B and fake-B distributions
              (distributional realism; standard GAN metric)
    - Modality-classifier "fool accuracy": a frozen CNN trained to separate
      real CT from real MRI is run on the *generated* images. The rate at
      which it mis-classifies fakes as the target modality is reported as an
      auxiliary "accuracy" figure — this is the closest well-defined notion of
      "accuracy" for an unsupervised image-translation task (there is no
      ground-truth label to compute classification accuracy against, since
      CT and MRI are unpaired). Report PSNR/SSIM/MAE/FID as your primary
      quantitative results; use this figure as a secondary/ablation metric.

Notes for the paper:
  * Compute all metrics on the held-out test split (testA/testB), never on
    images seen during training.
  * Report mean +/- 95% CI over the test set (see bootstrap_ci below), and a
    Wilcoxon signed-rank test against a CycleGAN baseline, exactly as done in
    the reference paper, to support a claim of statistical significance.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.stats import wilcoxon
import torchvision.models as tv_models


# --------------------------------------------------------------------------- #
# Pixel-level metrics
# --------------------------------------------------------------------------- #

def denorm(x):
    """[-1, 1] -> [0, 255] float, matching typical CT/MRI intensity reporting."""
    return ((x.clamp(-1, 1) + 1) / 2) * 255.0


def mae(fake, real):
    """Mean absolute error, in 0-255 intensity units (paper reports MAE ~ 19-20)."""
    return torch.mean(torch.abs(denorm(fake) - denorm(real))).item()


def psnr(fake, real, max_val=255.0):
    f, r = denorm(fake), denorm(real)
    mse = torch.mean((f - r) ** 2)
    if mse.item() == 0:
        return float("inf")
    return (20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)).item()


def _gaussian_window(size, sigma, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window = g.T @ g
    return window.unsqueeze(0).unsqueeze(0)


def ssim(fake, real, window_size=11, sigma=1.5):
    """Single-scale SSIM (Wang et al., 2004), computed per-image then averaged
    over the batch. Inputs expected in [-1, 1]."""
    device = fake.device
    window = _gaussian_window(window_size, sigma, device).to(fake.dtype)
    pad = window_size // 2

    f, r = denorm(fake), denorm(real)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    mu_f = F.conv2d(f, window, padding=pad)
    mu_r = F.conv2d(r, window, padding=pad)
    mu_f_sq, mu_r_sq, mu_fr = mu_f * mu_f, mu_r * mu_r, mu_f * mu_r

    sigma_f_sq = F.conv2d(f * f, window, padding=pad) - mu_f_sq
    sigma_r_sq = F.conv2d(r * r, window, padding=pad) - mu_r_sq
    sigma_fr = F.conv2d(f * r, window, padding=pad) - mu_fr

    ssim_map = ((2 * mu_fr + C1) * (2 * sigma_fr + C2)) / (
        (mu_f_sq + mu_r_sq + C1) * (sigma_f_sq + sigma_r_sq + C2)
    )
    return ssim_map.mean().item()


# --------------------------------------------------------------------------- #
# FID (distributional realism)
# --------------------------------------------------------------------------- #

class InceptionFeatureExtractor(torch.nn.Module):
    """Pool5 features from an ImageNet-pretrained Inception-v3, the standard
    backbone for FID. Grayscale slices are replicated to 3 channels."""

    def __init__(self, device):
        super().__init__()
        net = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1,
                                      aux_logits=True)
        net.fc = torch.nn.Identity()
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def __call__(self, x):
        x = (x.clamp(-1, 1) + 1) / 2
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.net(x).cpu().numpy()


def compute_fid(real_feats, fake_feats, eps=1e-6):
    mu1, mu2 = real_feats.mean(0), fake_feats.mean(0)
    sigma1, sigma2 = np.cov(real_feats, rowvar=False), np.cov(fake_feats, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


# --------------------------------------------------------------------------- #
# Modality-classifier "fool accuracy" (auxiliary metric — see module docstring)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def fool_accuracy(classifier, fake_images, target_is_mri):
    """Fraction of generated images the frozen real-CT-vs-real-MRI classifier
    assigns to the *intended target modality*. 50% = chance level (classifier
    can't tell), higher = more realistic in the classifier's eyes.
    NOT the same as supervised classification accuracy — there is no
    ground-truth label for an individual unpaired translated image; this
    measures distributional realism from the classifier's perspective."""
    logits = classifier(fake_images)
    preds_are_mri = (logits > 0)
    target = torch.ones_like(preds_are_mri) if target_is_mri else torch.zeros_like(preds_are_mri)
    return (preds_are_mri == target).float().mean().item()


# --------------------------------------------------------------------------- #
# Statistics for the paper (mean, 95% CI, significance vs. baseline)
# --------------------------------------------------------------------------- #

def bootstrap_ci(values, n_boot=10000, ci=95):
    values = np.asarray(values)
    boots = [np.mean(np.random.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(np.mean(values)), float(lo), float(hi)


def significance_vs_baseline(method_scores, baseline_scores):
    """Two-sided Wilcoxon signed-rank test, exactly as reported in Kearney et
    al. 2020 (Table of MAE/SSIM/PSNR p-values)."""
    stat, p_value = wilcoxon(method_scores, baseline_scores)
    return float(stat), float(p_value)
