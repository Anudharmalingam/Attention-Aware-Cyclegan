"""
config.py
Central configuration for Attention-Aware CycleGAN (A-CycleGAN).

*** PRIMARY TASK: CT -> MRI translation ***
    Domain A = CT   (input you actually want translated)
    Domain B = MRI  (target modality)
    The model you care about for the paper/deliverable is G_AB : CT -> MRI.
    G_BA : MRI -> CT is trained jointly (CycleGAN requires both directions for
    the cycle-consistency loss) and is a useful secondary result, but it is
    NOT the primary task -- don't get the two directions confused when
    reading logs/samples. train.py prints this mapping explicitly at startup.

Dataset reference : https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan
Method reference  : Kearney V, et al. "Attention-Aware Discrimination for MR-to-CT
                     Image Translation Using Cycle-Consistent Generative Adversarial
                     Networks." Radiol Artif Intell. 2020;2(2):e190027.
                     https://pubmed.ncbi.nlm.nih.gov/33937817/
                     (Original paper does MR->CT; here the same attention-aware,
                     VAE-regularised discriminator + cycle-consistency architecture is
                     used for CT->MRI, since CycleGAN training is bidirectional by design.)
"""

import os
import torch

class Config:
    # ---------------------------------------------------------------- paths
    DATA_ROOT   = r"D:\CT_MRI_A\Dataset\images"                  # expects trainA/trainB/testA/testB inside
    TRAIN_A_DIR = os.path.join(DATA_ROOT, "trainA")   # Domain A = CT  (translation source)
    TRAIN_B_DIR = os.path.join(DATA_ROOT, "trainB")   # Domain B = MRI (translation target)
    TEST_A_DIR  = os.path.join(DATA_ROOT, "testA")    # Domain A = CT
    TEST_B_DIR  = os.path.join(DATA_ROOT, "testB")    # Domain B = MRI

    CKPT_DIR    = "./checkpoints"
    RESULT_DIR  = "./results"
    LOG_DIR     = "./logs"

    # ------------------------------------------------------------- hardware
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    AMP         = True          # mixed precision (torch.cuda.amp) -> big speedup on NVIDIA GPUs
    SEED        = 42

    # ------------------------------------------------------------- image
    IMG_CHANNELS = 1            # grayscale CT/MRI
    IMG_SIZE     = 256          # resized from native 512x512 (raise if you have VRAM to spare)

    # ------------------------------------------------------------- model
    NGF               = 64      # generator base filters
    NDF               = 64      # discriminator base filters
    N_RES_BLOCKS      = 9       # ResNet blocks in generator (standard CycleGAN = 9 for 256px)
    USE_SELF_ATTENTION = True   # self-attention block inside generator bottleneck
    USE_VAE_DISC       = True   # variational-encoding enhancement on discriminator (paper's A-CycleGAN)
    ATTN_GATE_CHANNELS = 64     # channels used inside attention-gating module of discriminator

    # ------------------------------------------------------------- training
    BATCH_SIZE     = 4
    EPOCHS         = 200
    DECAY_EPOCH    = 100        # linear LR decay to 0 starts here (standard CycleGAN schedule)
    LR             = 2e-4       # kept for backward compatibility; LR_G below is what's actually used
    LR_G           = 2e-4       # generator learning rate
    LR_D           = 1e-4       # discriminator learning rate -- HALF of LR_G.
                                 # Fix for the "D wins too early / G stuck at identity mapping"
                                 # failure mode: the attention-gated + VAE discriminator is a
                                 # deeper, more expressive network than the ResNet generator's
                                 # per-step adversarial signal can keep up with at equal LR. If
                                 # D_A/D_B losses are still pinned near 0 for many epochs after
                                 # restarting with this change, lower LR_D further (e.g. 5e-5).
    BETA1          = 0.5
    BETA2          = 0.999

    # ------------------------------------------------------------- loss weights
    LAMBDA_CYCLE     = 10.0     # cycle-consistency weight
    LAMBDA_IDENTITY  = 1.5      # REDUCED from 5.0 -> 1.5. At the original weight, identity loss
                                 # dominates the (currently weak) adversarial signal and rewards
                                 # the generator for barely changing the image at all -- exactly
                                 # the identity-mapping failure you saw in the sample grids.
                                 # 1.5 still gives some colour/structure stability but no longer
                                 # drowns out the GAN loss. Once translation is visibly working,
                                 # you can raise it back towards 3-5 in a later fine-tuning pass
                                 # if outputs drift from source anatomy too much.
    LAMBDA_KL        = 0.01     # VAE KL-divergence weight on discriminator latent
    LAMBDA_PERCEPTUAL= 1.0      # VGG perceptual loss weight (set to 0.0 to disable)
    USE_PERCEPTUAL   = True

    GAN_LOSS_MODE       = "lsgan"  # 'lsgan' (MSE) is more stable than vanilla BCE for CycleGAN
    REAL_LABEL_SMOOTHING = 0.9     # NEW: discriminator's "real" target is 0.9 instead of 1.0.
                                    # One-sided label smoothing (Salimans et al., 2016) softens
                                    # D's confidence, slowing its convergence relative to G without
                                    # touching the "fake" target (still 0.0) -- a standard, safe
                                    # fix for a discriminator that's overpowering the generator.

    # ------------------------------------------------------------- buffers / logging
    REPLAY_BUFFER_SIZE = 50     # historical fake-image buffer to stabilise discriminator training
    SAMPLE_INTERVAL     = 200   # iterations between saved sample grids
    CKPT_INTERVAL        = 5    # epochs between checkpoint saves
    EVAL_INTERVAL         = 5   # epochs between full test-set metric evaluation

    # ------------------------------------------------------------- extra lever (optional)
    D_UPDATE_EVERY = 1          # update D every N generator steps (N=1 -> normal, every step).
                                 # If D_A/D_B are STILL pinned near 0 after the LR_D + label-
                                 # smoothing fix above, try D_UPDATE_EVERY=2 (skip every other D
                                 # update) to slow the discriminator down further without touching
                                 # loss weights again. Leave at 1 for the first restart.

cfg = Config()
