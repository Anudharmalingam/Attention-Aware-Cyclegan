"""
train.py
Train the Attention-Aware CycleGAN (A-CycleGAN).

*** PRIMARY TASK: CT -> MRI translation (G_AB). ***
    Domain A (trainA/testA) = CT, Domain B (trainB/testB) = MRI.
    G_BA (MRI -> CT) is trained jointly for cycle-consistency and is a useful
    secondary result, but G_AB is the model this project is built for.

Usage:
    python train.py                      # uses defaults in config.py
    python train.py --epochs 300 --batch_size 8

Requires an NVIDIA GPU for reasonable training time (uses torch.cuda.amp
mixed precision automatically when available).
"""

import argparse
import itertools
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from config import cfg
from datasets import UnpairedCTMRIDataset, ReplayBuffer
from models import AttentionResNetGenerator, AttentionVAEDiscriminator, weights_init_normal
from losses import CombinedLosses
from metrics import mae, psnr, ssim


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def lr_lambda(epoch, total_epochs, decay_epoch):
    if epoch < decay_epoch:
        return 1.0
    return max(0.0, 1.0 - (epoch - decay_epoch) / float(total_epochs - decay_epoch))


def build_models(device):
    G_AB = AttentionResNetGenerator(cfg.IMG_CHANNELS, cfg.IMG_CHANNELS, cfg.NGF,
                                     cfg.N_RES_BLOCKS, cfg.USE_SELF_ATTENTION).to(device)
    G_BA = AttentionResNetGenerator(cfg.IMG_CHANNELS, cfg.IMG_CHANNELS, cfg.NGF,
                                     cfg.N_RES_BLOCKS, cfg.USE_SELF_ATTENTION).to(device)
    D_A = AttentionVAEDiscriminator(cfg.IMG_CHANNELS, cfg.NDF, cfg.USE_VAE_DISC).to(device)
    D_B = AttentionVAEDiscriminator(cfg.IMG_CHANNELS, cfg.NDF, cfg.USE_VAE_DISC).to(device)
    for net in (G_AB, G_BA, D_A, D_B):
        net.apply(weights_init_normal)
    return G_AB, G_BA, D_A, D_B


@torch.no_grad()
def evaluate(G_AB, G_BA, loader, device, max_batches=None):
    """Compute MAE / PSNR / SSIM on a held-out loader for both translation
    directions. Returns dict of per-metric lists (so you can bootstrap CIs).
    Keys: CT2MRI_* = cycle CT->MRI->CT reconstruction fidelity (proxy for the
    primary G_AB task); MRI2CT_* = cycle MRI->CT->MRI (secondary G_BA task)."""
    G_AB.eval(); G_BA.eval()
    out = {"CT2MRI_MAE": [], "CT2MRI_PSNR": [], "CT2MRI_SSIM": [],
           "MRI2CT_MAE": [], "MRI2CT_PSNR": [], "MRI2CT_SSIM": []}
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        real_A, real_B = batch["A"].to(device), batch["B"].to(device)   # A=CT, B=MRI
        fake_B = G_AB(real_A)     # CT -> MRI  (primary task)
        fake_A = G_BA(real_B)     # MRI -> CT  (secondary task)
        rec_A = G_BA(fake_B)      # CT -> MRI -> CT
        rec_B = G_AB(fake_A)      # MRI -> CT -> MRI
        # cycle-consistency proxy quality (no paired ground truth exists between CT and
        # MRI directly, so we report reconstruction fidelity, standard practice for
        # unpaired CT/MRI translation papers where no paired ground truth is available).
        out["CT2MRI_MAE"].append(mae(rec_A, real_A))
        out["CT2MRI_PSNR"].append(psnr(rec_A, real_A))
        out["CT2MRI_SSIM"].append(ssim(rec_A, real_A))
        out["MRI2CT_MAE"].append(mae(rec_B, real_B))
        out["MRI2CT_PSNR"].append(psnr(rec_B, real_B))
        out["MRI2CT_SSIM"].append(ssim(rec_B, real_B))
    G_AB.train(); G_BA.train()
    return out


def main(args):
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print("=" * 70)
    print("TASK: CT -> MRI translation  (G_AB : Domain A[CT] -> Domain B[MRI])")
    print(f"  Domain A (CT)  train dir: {cfg.TRAIN_A_DIR}")
    print(f"  Domain B (MRI) train dir: {cfg.TRAIN_B_DIR}")
    print("  G_BA (MRI -> CT) is also trained, for cycle-consistency only.")
    print("=" * 70)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"LR_G={args.lr_g}  LR_D={args.lr_d}  LAMBDA_IDENTITY={cfg.LAMBDA_IDENTITY}  "
          f"REAL_LABEL_SMOOTHING={cfg.REAL_LABEL_SMOOTHING}  D_UPDATE_EVERY={cfg.D_UPDATE_EVERY}")

    train_set = UnpairedCTMRIDataset(cfg.TRAIN_A_DIR, cfg.TRAIN_B_DIR, cfg.IMG_SIZE, train=True,
                                      channels=cfg.IMG_CHANNELS)
    test_set = UnpairedCTMRIDataset(cfg.TEST_A_DIR, cfg.TEST_B_DIR, cfg.IMG_SIZE, train=False,
                                     channels=cfg.IMG_CHANNELS)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True)

    G_AB, G_BA, D_A, D_B = build_models(device)
    losses = CombinedLosses(cfg, device)

    opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                              lr=args.lr_g, betas=(cfg.BETA1, cfg.BETA2))
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr_d, betas=(cfg.BETA1, cfg.BETA2))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr_d, betas=(cfg.BETA1, cfg.BETA2))

    sched_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G, lr_lambda=lambda e: lr_lambda(e, args.epochs, cfg.DECAY_EPOCH))
    sched_D_A = torch.optim.lr_scheduler.LambdaLR(
        opt_D_A, lr_lambda=lambda e: lr_lambda(e, args.epochs, cfg.DECAY_EPOCH))
    sched_D_B = torch.optim.lr_scheduler.LambdaLR(
        opt_D_B, lr_lambda=lambda e: lr_lambda(e, args.epochs, cfg.DECAY_EPOCH))

    fake_A_buffer, fake_B_buffer = ReplayBuffer(cfg.REPLAY_BUFFER_SIZE), ReplayBuffer(cfg.REPLAY_BUFFER_SIZE)
    use_amp = cfg.AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(cfg.LOG_DIR)

    step = 0
    best_ssim = -1.0
    d_A_breakdown, d_B_breakdown = {"D_total": 0.0}, {"D_total": 0.0}  # placeholders until first D update
    for epoch in range(args.epochs):
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            # ---------------------- Train Generators ----------------------
            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                idt_A = G_BA(real_A)     # identity: BA(A) ~ A
                idt_B = G_AB(real_B)     # identity: AB(B) ~ B

                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)
                rec_A = G_BA(fake_B)
                rec_B = G_AB(fake_A)

                pred_fake_B, _ = D_B(fake_B)
                pred_fake_A, _ = D_A(fake_A)

                g_total, g_breakdown = losses.generator_loss(
                    real_A, real_B, fake_A, fake_B, rec_A, rec_B, idt_A, idt_B,
                    pred_fake_A, pred_fake_B)

            scaler.scale(g_total).backward()
            scaler.step(opt_G)

            # D_UPDATE_EVERY > 1 lets you skip discriminator updates on some steps if it's
            # still overpowering the generator after the LR_D + label-smoothing fix (see
            # config.py). Default is 1 (update every step, i.e. normal CycleGAN behaviour).
            update_d_this_step = (step % cfg.D_UPDATE_EVERY == 0)

            # ---------------------- Train Discriminator A ----------------------
            if update_d_this_step:
                opt_D_A.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_real_A, extras_real_A = D_A(real_A)
                    fake_A_ = fake_A_buffer.push_and_pop(fake_A.detach())
                    pred_fake_A_, _ = D_A(fake_A_)
                    d_A_loss, d_A_breakdown = losses.discriminator_loss(
                        pred_real_A, pred_fake_A_, extras_real_A["mu"], extras_real_A["logvar"])
                scaler.scale(d_A_loss).backward()
                scaler.step(opt_D_A)

            # ---------------------- Train Discriminator B ----------------------
            if update_d_this_step:
                opt_D_B.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_real_B, extras_real_B = D_B(real_B)
                    fake_B_ = fake_B_buffer.push_and_pop(fake_B.detach())
                    pred_fake_B_, _ = D_B(fake_B_)
                    d_B_loss, d_B_breakdown = losses.discriminator_loss(
                        pred_real_B, pred_fake_B_, extras_real_B["mu"], extras_real_B["logvar"])
                scaler.scale(d_B_loss).backward()
                scaler.step(opt_D_B)

            scaler.update()

            if step % 50 == 0:
                print(f"[Epoch {epoch}/{args.epochs}][Batch {i}/{len(train_loader)}] "
                      f"G: {g_breakdown['G_total']:.3f} | D_A: {d_A_breakdown['D_total']:.3f} | "
                      f"D_B: {d_B_breakdown['D_total']:.3f}")
                for k, v in {**g_breakdown, **d_A_breakdown, **d_B_breakdown}.items():
                    writer.add_scalar(f"loss/{k}", v, step)

            if step % cfg.SAMPLE_INTERVAL == 0:
                with torch.no_grad():
                    # Grid rows: real CT | fake MRI (G_AB output, the primary result) |
                    #            real MRI | fake CT (G_BA output, secondary)
                    grid = make_grid(torch.cat([real_A[:4], fake_B[:4], real_B[:4], fake_A[:4]]),
                                      nrow=4, normalize=True)
                    save_image(grid, os.path.join(cfg.RESULT_DIR, f"sample_step{step}.png"))
            step += 1

        sched_G.step(); sched_D_A.step(); sched_D_B.step()
        print(f"Epoch {epoch} finished in {time.time() - t0:.1f}s "
              f"| lr={opt_G.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % cfg.EVAL_INTERVAL == 0 or epoch == args.epochs - 1:
            metrics = evaluate(G_AB, G_BA, test_loader, device, max_batches=50)
            mean_ssim = float(np.mean(metrics["CT2MRI_SSIM"] + metrics["MRI2CT_SSIM"]))
            print(f"  [Eval] CT->MRI (primary) MAE={np.mean(metrics['CT2MRI_MAE']):.3f} "
                  f"PSNR={np.mean(metrics['CT2MRI_PSNR']):.3f} SSIM={np.mean(metrics['CT2MRI_SSIM']):.4f} | "
                  f"MRI->CT (secondary) MAE={np.mean(metrics['MRI2CT_MAE']):.3f} "
                  f"PSNR={np.mean(metrics['MRI2CT_PSNR']):.3f} SSIM={np.mean(metrics['MRI2CT_SSIM']):.4f}")
            for k, v in metrics.items():
                writer.add_scalar(f"eval/{k}", float(np.mean(v)), epoch)
            if mean_ssim > best_ssim:
                best_ssim = mean_ssim
                torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                            "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                            "epoch": epoch, "ssim": best_ssim},
                           os.path.join(cfg.CKPT_DIR, "best.pth"))

        if (epoch + 1) % cfg.CKPT_INTERVAL == 0:
            torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                        "D_A": D_A.state_dict(), "D_B": D_B.state_dict(), "epoch": epoch},
                       os.path.join(cfg.CKPT_DIR, f"epoch_{epoch+1}.pth"))

    writer.close()
    print("Training complete. Best SSIM:", best_ssim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr_g", type=float, default=cfg.LR_G, help="generator learning rate")
    parser.add_argument("--lr_d", type=float, default=cfg.LR_D, help="discriminator learning rate")
    args = parser.parse_args()
    main(args)
