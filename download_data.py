"""
download_data.py
Downloads and unpacks the Kaggle "CT-to-MRI-CGAN" dataset
(https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan) into ./data/
with the trainA/trainB/testA/testB layout expected by datasets.py.

Prerequisite: a Kaggle API token.
    1. Go to https://www.kaggle.com/settings -> "Create New Token"
    2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json  (chmod 600)
    3. pip install kaggle

Usage:
    python download_data.py
"""

import os
import subprocess
import zipfile
import shutil
import glob

DATASET = "darren2020/ct-to-mri-cgan"
DEST = "./data"


def main():
    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading {DATASET} via Kaggle API ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", DEST], check=True)

    zips = glob.glob(os.path.join(DEST, "*.zip"))
    if not zips:
        raise RuntimeError("No zip file downloaded — check your Kaggle API credentials.")

    for z in zips:
        print(f"Extracting {z} ...")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(DEST)
        os.remove(z)

    # The archive sometimes nests everything under an extra folder; flatten if needed.
    for split in ["trainA", "trainB", "testA", "testB"]:
        matches = glob.glob(os.path.join(DEST, "**", split), recursive=True)
        target = os.path.join(DEST, split)
        if matches and not os.path.isdir(target):
            shutil.move(matches[0], target)

    for split in ["trainA", "trainB", "testA", "testB"]:
        p = os.path.join(DEST, split)
        n = len(os.listdir(p)) if os.path.isdir(p) else 0
        print(f"  {split}: {n} images {'OK' if n > 0 else 'MISSING - check extraction'}")

    print("Done. Dataset ready under ./data/")


if __name__ == "__main__":
    main()
