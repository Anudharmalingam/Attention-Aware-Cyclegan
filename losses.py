"""
losses.py
Loss functions for A-CycleGAN CT<->MRI training.

Total objective (per direction, summed over A->B and B->A):

    L = L_adv(G, D)                         # LSGAN adversarial loss
      + lambda_cycle    * L_cycle           # ||G_BA(G_AB(A)) - A||_1  (and symmetric)
      + lambda_identity * L_identity         # ||G_AB(B) - B||_1        (and symmetric)
      + lambda_kl       * L_kl               # KL(q(z|x) || N(0,I)) on discriminator VAE head
      + lambda_perc     * L_perceptual        # VGG feature loss between real and reconstructed

This combination follows standard CycleGAN (Zhu et al., 2017) for the
generator terms, and adds the KL term used by the attention-aware, VAE-
enhanced discriminator of Kearney et al., 2020
(https://pubmed.ncbi.nlm.nih.gov/33937817/).
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class GANLoss(nn.Module):
    """LSGAN (least-squares) loss — empirically more stable than vanilla
    BCE-based GAN loss for CycleGAN-style training, and standard in the field.

    Supports one-sided label smoothing (Salimans et al., 2016): the "real"
    target is real_label (e.g. 0.9) instead of 1.0, which softens the
    discriminator's confidence and slows its convergence relative to the
    generator -- a standard fix when D overpowers G early in training. The
    "fake" target is left at 0.0 (two-sided smoothing can encourage the
    generator to produce blurry/washed-out outputs, so we avoid it here)."""

    def __init__(self, mode="lsgan", real_label=1.0):
        super().__init__()
        self.mode = mode
        self.real_label = real_label
        if mode == "lsgan":
            self.loss = nn.MSELoss()
        elif mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(mode)

    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target = torch.full_like(prediction, self.real_label)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)


def kl_divergence(mu, logvar):
    """Closed-form KL divergence between N(mu, sigma^2) and N(0, I), averaged
    over the batch (standard VAE regulariser)."""
    if mu is None or logvar is None:
        return torch.tensor(0.0)
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class VGGPerceptualLoss(nn.Module):
    """Perceptual (feature-space) loss using a frozen, ImageNet-pretrained VGG16.
    Grayscale CT/MRI slices are replicated to 3 channels since VGG expects RGB.
    This is optional (Config.USE_PERCEPTUAL) — some medical-imaging papers omit
    it because VGG features are natural-image-domain; it is offered here as it
    often sharpens fine anatomical detail and is commonly reported as an ablation."""

    def __init__(self, layer_idx=16):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features[:layer_idx]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()
        self.criterion = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):
        # x is in [-1, 1], single channel -> [0,1], 3-channel, ImageNet-normalised
        x = (x + 1) / 2
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        f = self.vgg(self._prep(fake))
        r = self.vgg(self._prep(real))
        return self.criterion(f, r.detach())


class CombinedLosses:
    """Convenience container bundling all loss terms + their configured weights."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        # Generator's adversarial target stays at full strength (1.0) -- it should always
        # aim to fully fool D for the cleanest possible gradient. Only the discriminator's
        # own "real" target is smoothed (see GANLoss docstring / config.py REAL_LABEL_SMOOTHING).
        self.gan_loss_G = GANLoss(cfg.GAN_LOSS_MODE, real_label=1.0).to(device)
        self.gan_loss_D = GANLoss(cfg.GAN_LOSS_MODE, real_label=getattr(cfg, "REAL_LABEL_SMOOTHING", 1.0)).to(device)
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss().to(device) if cfg.USE_PERCEPTUAL else None

    def generator_loss(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B,
                        idt_A, idt_B, pred_fake_A, pred_fake_B):
        cfg = self.cfg
        loss_gan_AB = self.gan_loss_G(pred_fake_B, True)   # G_AB fools D_B
        loss_gan_BA = self.gan_loss_G(pred_fake_A, True)   # G_BA fools D_A

        loss_cycle_A = self.cycle_loss(rec_A, real_A) * cfg.LAMBDA_CYCLE
        loss_cycle_B = self.cycle_loss(rec_B, real_B) * cfg.LAMBDA_CYCLE

        loss_idt_A = self.identity_loss(idt_A, real_A) * cfg.LAMBDA_IDENTITY
        loss_idt_B = self.identity_loss(idt_B, real_B) * cfg.LAMBDA_IDENTITY

        loss_perc = torch.tensor(0.0, device=real_A.device)
        if self.perceptual_loss is not None:
            loss_perc = (self.perceptual_loss(fake_B, real_A) +
                         self.perceptual_loss(fake_A, real_B)) * cfg.LAMBDA_PERCEPTUAL

        total = (loss_gan_AB + loss_gan_BA + loss_cycle_A + loss_cycle_B +
                 loss_idt_A + loss_idt_B + loss_perc)

        breakdown = {
            "G_total": total.item(),
            "G_gan_AB": loss_gan_AB.item(), "G_gan_BA": loss_gan_BA.item(),
            "G_cycle_A": loss_cycle_A.item(), "G_cycle_B": loss_cycle_B.item(),
            "G_idt_A": loss_idt_A.item(), "G_idt_B": loss_idt_B.item(),
            "G_perceptual": loss_perc.item() if torch.is_tensor(loss_perc) else loss_perc,
        }
        return total, breakdown

    def discriminator_loss(self, pred_real, pred_fake, mu=None, logvar=None):
        loss_real = self.gan_loss_D(pred_real, True)
        loss_fake = self.gan_loss_D(pred_fake, False)
        loss_kl = kl_divergence(mu, logvar).to(pred_real.device) * self.cfg.LAMBDA_KL
        total = 0.5 * (loss_real + loss_fake) + loss_kl
        breakdown = {
            "D_total": total.item(), "D_real": loss_real.item(),
            "D_fake": loss_fake.item(), "D_kl": float(loss_kl.detach()),
        }
        return total, breakdown
