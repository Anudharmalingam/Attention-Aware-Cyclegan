# CT-to-MRI Translation with Attention-Aware CycleGAN

## Project summary

This project implements an unpaired medical image translation system for CT-to-MRI conversion using a customized CycleGAN-based architecture. The main objective is to learn a mapping from CT domain images to MRI domain images without paired training data.

The implementation is based on the standard CycleGAN formulation of Zhu et al. (2017), but it extends the architecture with attention-aware discriminator components and a variational latent regularization term, following the paper by Kearney et al. on attention-aware discrimination for MR-to-CT translation.

In this project, the primary task is:

- CT -> MRI: G_AB
- MRI -> CT: G_BA (used jointly to enforce cycle-consistency)

The model is trained on unpaired grayscale CT and MRI slices from the Kaggle CT-to-MRI-CGAN dataset.

---

## 1. Problem definition

This is a domain translation problem:

- Domain A = CT images
- Domain B = MRI images
- No paired CT/MRI example exists for the same anatomy
- The model must learn a mapping in an unsupervised/unpaired setting

Because paired ground truth is unavailable, the project uses cycle-consistency instead of direct supervised pixel supervision.

---

## 2. Core methodology used in the project

### 2.1 CycleGAN backbone

The fundamental algorithm is CycleGAN.

CycleGAN learns two generators:

- G_AB : CT -> MRI
- G_BA : MRI -> CT

and two discriminators:

- D_A : distinguishes real CT from fake CT
- D_B : distinguishes real MRI from fake MRI

The generator is trained to produce realistic outputs that fool the discriminator, while the cycle-consistency constraint keeps the generated images anatomically consistent with the original input.

The key idea is:

- A CT image should be translated to MRI
- The MRI image should be able to be translated back to CT
- The reconstructed image should remain similar to the original image

This is the core reason CycleGAN is useful for medical imaging when direct paired labels do not exist.

---

### 2.2 Generator architecture

The generator used in the project is a ResNet-based encoder-decoder structure, similar to CycleGAN.

#### Generator design

- Reflection padding
- Initial convolution
- Downsampling blocks
- Residual blocks
- Self-attention block in the bottleneck
- Upsampling blocks
- Final Tanh output

#### Why self-attention is used

The generator includes a Self-Attention (SAGAN-style) block in the bottleneck.

This helps the network focus on long-range anatomical relationships rather than local patches only. For medical images, this is important because structures like:

- brain anatomy
- skull boundaries
- ventricles
- tissue contrast patterns

can be spatially distributed across the image, and a purely local convolutional network may struggle to preserve those relationships.

---

### 2.3 Discriminator architecture

The discriminator is not a simple PatchGAN discriminator. It is an attention-aware, VAE-enhanced discriminator.

#### Main discriminator components

- PatchGAN-style convolutional block
- Multi-scale feature extraction
- Attention gating between feature maps
- Optional VAE head for latent-space regularization

#### Attention gate

The project implements an attention gate that takes:

- shallow feature maps (skip path)
- deeper semantic feature maps (gating signal)

and reweights the shallow features based on the deeper features.

This is the main idea of the paper's attention-aware discrimination:

- important anatomical regions are emphasized
- less relevant regions are suppressed
- discriminator focuses on salient structures instead of all pixels equally

This is especially useful in medical imaging where some regions have more clinical importance than others.

#### VAE head

A VAE-style latent branch is added to the discriminator bottleneck.

It predicts:

- mu
- logvar

and uses the standard VAE reparameterization trick.

This helps regularize the latent space of the discriminator and stabilizes training when using a deeper discriminator.

---

## 3. Loss functions

The training objective combines multiple loss terms.

### 3.1 Adversarial loss

The model uses Least Squares GAN (LSGAN) instead of vanilla BCE GAN loss.

Why LSGAN was chosen:

- more stable for CycleGAN training
- reduces vanishing gradient problems in some cases
- often gives smoother convergence for image translation tasks

The discriminator attempts to classify real and fake images correctly, while the generator tries to fool it.

---

### 3.2 Cycle-consistency loss

This is the central component of CycleGAN.

For CT -> MRI -> CT reconstruction:

- G_BA(G_AB(A)) ≈ A

For MRI -> CT -> MRI reconstruction:

- G_AB(G_BA(B)) ≈ B

The project uses L1 loss for this constraint:

- L_cycle = ||G_BA(G_AB(A)) - A||_1
- L_cycle = ||G_AB(G_BA(B)) - B||_1

This ensures the translation is invertible enough to preserve the original content.

---

### 3.3 Identity loss

The model also includes identity mapping regularization:

- G_BA(A) ≈ A
- G_AB(B) ≈ B

This means if an image is already in the target domain, the generator should not significantly change it.

This helps preserve content when the domains are visually similar in structure but differ in appearance or intensity.

---

### 3.4 KL divergence loss from VAE branch

The discriminator's VAE head contributes KL divergence regularization:

- KL(q(z|x) || N(0, I))

This keeps the discriminator latent space stable and prevents overfitting or exploding latent variance.

---

### 3.5 Perceptual loss

The project optionally uses a VGG16-based perceptual loss.

This compares deep layer features between a generated image and a real image rather than only comparing raw pixels.

This helps preserve:

- anatomy
- texture
- structures
- semantic consistency

The perceptual loss is especially useful when the translated images need to look realistic rather than just numerically similar.

---

### 3.6 Combined objective

The total generator loss is approximately:

- adversarial loss
- + cycle-consistency loss
- + identity loss
- + perceptual loss

The discriminator loss includes:

- real-image loss
- fake-image loss
- KL regularization term

The actual project code implements this in [losses.py](losses.py).

---

## 4. Training algorithm used in this project

The training algorithm is as follows:

1. Load CT and MRI images from unpaired training directories
2. Create paired batch samples by random domain sampling from each domain
3. Forward CT images through G_AB to generate fake MRI
4. Forward MRI images through G_BA to generate fake CT
5. Reconstruct back to original domains using cycle passes
6. Compute generator loss using adversarial + cycle + identity + optional perceptual terms
7. Update the generator parameters
8. Train the discriminator on real/fake CT and MRI samples
9. Use a replay buffer for fake images to stabilize training
10. Evaluate model periodically on held-out test data
11. Save the best model based on SSIM performance

This is implemented in [train.py](train.py).

---

## 5. Important stabilizing tricks used in the implementation

This project is not a plain vanilla CycleGAN. It includes several practical stabilizers that are widely used in modern GAN training.

### 5.1 LSGAN instead of BCE

Least Squares GAN is more stable for image-to-image translation tasks and often reduces mode collapse or overly aggressive discriminator behavior.

### 5.2 Label smoothing

The discriminator’s target for real samples is softened from 1.0 to 0.9.

This helps prevent the discriminator from becoming too confident too early, which can suppress generator learning.

### 5.3 Lower discriminator learning rate

The discriminator learning rate is intentionally set lower than the generator learning rate.

This prevents the discriminator from overpowering the generator in early training.

### 5.4 Replay buffer

The project stores a history of generated fake images and uses them for discriminator training instead of always training on the current batch only.

This reduces oscillation and stabilizes adversarial training.

### 5.5 Mixed precision training

If CUDA is available, the project uses mixed precision (AMP) to speed up training and reduce memory usage.

### 5.6 Lambda learning-rate decay

The training schedule decays the learning rate over time using a linear schedule.

This helps convergence and avoids large final oscillations.

---

## 6. Dataset and preprocessing

The dataset is stored in a folder structure like:

- trainA/ for CT images
- trainB/ for MRI images
- testA/ for CT validation/test images
- testB/ for MRI validation/test images

Preprocessing steps:

- resize to 256x256
- random horizontal flip during training
- small random rotation during training
- convert to tensor
- normalize to [-1, 1]

This is done in [datasets.py](datasets.py).

The output is suitable for a generator that ends with Tanh activation.

---

## 7. Evaluation metrics used

The project evaluates both image fidelity and distribution realism.

### Pixel-level metrics

- MAE
- PSNR
- SSIM

These measure how close generated images are to the target content in intensity and structure.

### Distributional realism metric

- FID (Fréchet Inception Distance)

This compares the feature distribution of real and generated images.

### Auxiliary realism metric

- modality-classifier fool accuracy

This measures whether a trained modality classifier is fooled by generated images into believing they are real target-domain images.

These metrics are defined in [metrics.py](metrics.py).

---

## 8. Why this method is appropriate for this task

This project is a strong fit for CT-to-MRI translation because:

- CT and MRI are different image domains
- paired data is unavailable
- the tasks are highly anatomical and structured
- CycleGAN is a standard approach for unpaired medical domain translation
- attention-aware discrimination helps focus on anatomically important regions
- VAE regularization improves stability of the discriminator
- identity and perceptual losses improve realism and content preservation

This makes the method more suitable than a simple vanilla GAN for medical image translation because it balances realism, structure preservation, and training stability.

---

## 9. Mathematical interpretation of the objective

The generator tries to minimize:

L_G = L_adv + λ_cycle * L_cycle + λ_identity * L_identity + λ_perc * L_perc

where:

- L_adv = adversarial loss to fool the target discriminator
- L_cycle = cycle-consistency reconstruction loss
- L_identity = identity loss
- L_perc = perceptual loss

The discriminator minimizes:

L_D = 0.5 * (L_real + L_fake) + λ_kl * L_KL

where:

- L_real = loss on real samples
- L_fake = loss on fake samples
- L_KL = KL regularization from VAE latent space

This is exactly the structure implemented in [losses.py](losses.py).

---

## 10. Files in this project

- [config.py](config.py) — all hyperparameters and settings
- [datasets.py](datasets.py) — data loading and preprocessing
- [models.py](models.py) — generator and discriminator design
- [losses.py](losses.py) — all training losses
- [train.py](train.py) — training loop and optimization process
- [metrics.py](metrics.py) — evaluation pipeline
- [test.py](test.py) — testing/inference utilities
- [download_data.py](download_data.py) — dataset download helper

---

## 11. Training command

To train the model:

```bash
python train.py
```

Optional custom training configuration:

```bash
python train.py --epochs 200 --batch_size 4 --lr_g 2e-4 --lr_d 1e-4
```

---

## 12. Final analysis

This project uses a modern, stable, and clinically motivated variant of CycleGAN for CT-to-MRI translation. The method combines:

- CycleGAN framework
- ResNet generators
- self-attention mechanism
- PatchGAN discriminator
- attention-gated discriminator
- VAE latent regularization
- LSGAN adversarial learning
- cycle consistency
- identity consistency
- perceptual loss
- replay buffer and label smoothing for stability

This is not a simple baseline GAN model; it is a carefully engineered medical image translation pipeline designed to improve anatomical consistency, realism, and training stability in an unpaired setting.

In short, the algorithm is:

"An attention-aware CycleGAN with LSGAN adversarial loss, cycle-consistency and identity regularization, and discriminator VAE stabilization for unpaired CT-to-MRI image translation."

---

## 13. Conclusion

The project implements a robust medical image translation method tailored for unpaired CT/MRI conversion. The combination of CycleGAN with attention-aware discrimination and VAE regularization makes it more suitable than a plain vanilla CycleGAN for medical imaging tasks where structural fidelity matters.
