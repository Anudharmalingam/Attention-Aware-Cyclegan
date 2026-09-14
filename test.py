 
"""
test.py
Final held-out evaluation, producing the numbers you report in the paper:
MAE, PSNR, SSIM (mean + 95% CI), FID, and the modality-classifier fool-rate.
 
Also trains (once) the small frozen CT-vs-MRI classifier used for the
fool-rate metric, and optionally runs a Wilcoxon signed-rank test against a
plain-CycleGAN checkpoint (baseline) if you provide one, mirroring the
statistical reporting style of Kearney et al. 2020.
 
Usage:
    python test.py --ckpt checkpoints/best.pth
    python test.py --ckpt checkpoints/best.pth --baseline_ckpt checkpoints/cyclegan_baseline.pth
"""
 
import argparse
import json
import os
 
import numpy as np
import torch
from torch.utils.data import DataLoader
 
from config import cfg
from datasets import UnpairedCTMRIDataset
from models import AttentionResNetGenerator, ModalityClassifier
from metrics import (mae, psnr, ssim, InceptionFeatureExtractor, compute_fid,
                      fool_accuracy, bootstrap_ci, significance_vs_baseline)
 
 
def load_generators(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    G_AB = AttentionResNetGenerator(cfg.IMG_CHANNELS, cfg.IMG_CHANNELS, cfg.NGF,
                                     cfg.N_RES_BLOCKS, cfg.USE_SELF_ATTENTION).to(device)
    G_BA = AttentionResNetGenerator(cfg.IMG_CHANNELS, cfg.IMG_CHANNELS, cfg.NGF,
                                     cfg.N_RES_BLOCKS, cfg.USE_SELF_ATTENTION).to(device)
    G_AB.load_state_dict(ck["G_AB"]); G_BA.load_state_dict(ck["G_BA"])
    G_AB.eval(); G_BA.eval()
    return G_AB, G_BA
 
 
def train_modality_classifier(train_loader, device, epochs=5):
    """Quick supervised classifier: real CT (label 0) vs real MRI (label 1).
    Used ONLY for the auxiliary fool-rate metric — frozen afterwards."""
    clf = ModalityClassifier(cfg.IMG_CHANNELS).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    clf.train()
    for ep in range(epochs):
        losses = []
        for batch in train_loader:
            real_A, real_B = batch["A"].to(device), batch["B"].to(device)
            x = torch.cat([real_A, real_B], dim=0)
            y = torch.cat([torch.zeros(real_A.size(0)), torch.ones(real_B.size(0))]).to(device)
            opt.zero_grad()
            logits = clf(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"  [classifier] epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")
    clf.eval()
    return clf
 
 
@torch.no_grad()
def collect_translations(G_AB, G_BA, loader, device):
    real_As, real_Bs, fake_Bs, fake_As, rec_As, rec_Bs = [], [], [], [], [], []
    for batch in loader:
        real_A, real_B = batch["A"].to(device), batch["B"].to(device)
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)
        real_As.append(real_A.cpu()); real_Bs.append(real_B.cpu())
        fake_Bs.append(fake_B.cpu()); fake_As.append(fake_A.cpu())
        rec_As.append(rec_A.cpu()); rec_Bs.append(rec_B.cpu())
    cat = lambda lst: torch.cat(lst, dim=0)
    return (cat(real_As), cat(real_Bs), cat(fake_Bs), cat(fake_As), cat(rec_As), cat(rec_Bs))
 
 
def per_sample_metric(fn, fake, real):
    return [fn(fake[i:i+1], real[i:i+1]) for i in range(fake.shape[0])]
 
 
def main(args):
    device = cfg.DEVICE
    test_set = UnpairedCTMRIDataset(cfg.TEST_A_DIR, cfg.TEST_B_DIR, cfg.IMG_SIZE, train=False,
                                     channels=cfg.IMG_CHANNELS)
    train_set = UnpairedCTMRIDataset(cfg.TRAIN_A_DIR, cfg.TRAIN_B_DIR, cfg.IMG_SIZE, train=True,
                                      channels=cfg.IMG_CHANNELS)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=cfg.NUM_WORKERS)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=cfg.NUM_WORKERS)
 
    G_AB, G_BA = load_generators(args.ckpt, device)
    real_A, real_B, fake_B, fake_A, rec_A, rec_B = collect_translations(G_AB, G_BA, test_loader, device)
 
    # -------- pixel metrics (reconstruction fidelity, the standard proxy when
    # no paired ground truth exists between CT and MRI domains) --------
    results = {}
    for name, fake, real in [("A2B_cycle(A->B->A)", rec_A, real_A), ("B2A_cycle(B->A->B)", rec_B, real_B)]:
        mae_list = per_sample_metric(mae, fake, real)
        psnr_list = per_sample_metric(psnr, fake, real)
        ssim_list = per_sample_metric(ssim, fake, real)
        results[name] = {
            "MAE": bootstrap_ci(mae_list),
            "PSNR": bootstrap_ci(psnr_list),
            "SSIM": bootstrap_ci(ssim_list),
            "_raw_ssim": ssim_list,  # kept for optional significance test below
        }
 
    # -------- FID (distributional realism of translated images) --------
    print("Computing FID (Inception-v3 features)...")
    incep = InceptionFeatureExtractor(device)
    real_B_feats = np.concatenate([incep(real_B[i:i+16].to(device)) for i in range(0, len(real_B), 16)])
    fake_B_feats = np.concatenate([incep(fake_B[i:i+16].to(device)) for i in range(0, len(fake_B), 16)])
    real_A_feats = np.concatenate([incep(real_A[i:i+16].to(device)) for i in range(0, len(real_A), 16)])
    fake_A_feats = np.concatenate([incep(fake_A[i:i+16].to(device)) for i in range(0, len(fake_A), 16)])
    results["FID_A2B(real_B_vs_fake_B)"] = compute_fid(real_B_feats, fake_B_feats)
    results["FID_B2A(real_A_vs_fake_A)"] = compute_fid(real_A_feats, fake_A_feats)
 
    # -------- modality-classifier fool accuracy (auxiliary metric) --------
    print("Training auxiliary CT-vs-MRI classifier for fool-rate metric...")
    clf = train_modality_classifier(train_loader, device, epochs=args.clf_epochs)
    results["fool_accuracy_A2B(fake_B_as_MRI)"] = fool_accuracy(clf, fake_B.to(device), target_is_mri=True)
    results["fool_accuracy_B2A(fake_A_as_CT)"] = fool_accuracy(clf, fake_A.to(device), target_is_mri=False)
 
    # -------- optional significance test vs a baseline checkpoint --------
    if args.baseline_ckpt:
        print("Evaluating baseline checkpoint for Wilcoxon significance test...")
        G_AB_b, G_BA_b = load_generators(args.baseline_ckpt, device)
        _, _, _, _, rec_A_b, rec_B_b = collect_translations(G_AB_b, G_BA_b, test_loader, device)
        base_ssim_A = per_sample_metric(ssim, rec_A_b, real_A)
        stat, p = significance_vs_baseline(results["A2B_cycle(A->B->A)"]["_raw_ssim"], base_ssim_A)
        results["wilcoxon_vs_baseline_SSIM_A2B"] = {"statistic": stat, "p_value": p}
 
    for k in list(results.keys()):
        if isinstance(results[k], dict) and "_raw_ssim" in results[k]:
            del results[k]["_raw_ssim"]
 
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    out_path = os.path.join(cfg.RESULT_DIR, "final_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
 
    print("\n===================== FINAL TEST-SET METRICS =====================")
    print(json.dumps(results, indent=2, default=str))
    print(f"\nSaved to {out_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to trained checkpoint (.pth)")
    parser.add_argument("--baseline_ckpt", type=str, default=None,
                         help="optional plain-CycleGAN checkpoint for significance comparison")
    parser.add_argument("--clf_epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)
 