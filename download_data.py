import os
import shutil
import kagglehub

COMPETITION = "store-sales-time-series-forecasting"
DEST = os.path.join("data", "raw")

os.makedirs(DEST, exist_ok=True)

print("Downloading competition data...")
path = kagglehub.competition_download(COMPETITION)
print(f"Downloaded to: {path}")

for fname in os.listdir(path):
    src = os.path.join(path, fname)
    dst = os.path.join(DEST, fname)
    shutil.copy2(src, dst)
    print(f"  Copied {fname}")

print(f"\nDone. Files are in '{DEST}/'")
