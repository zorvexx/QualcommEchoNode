import shutil
import os

src = r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv"
dst = r"data/raw/unified_dataset_14col.csv"

os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy(src, dst)
print(f"Copied {src} -> {dst} (Size: {os.path.getsize(dst)} bytes)")
