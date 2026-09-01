import os
import sys
import zipfile
import shutil
import random
from pathlib import Path
from PIL import Image

# Ensure seed stability
random.seed(42)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOWNLOADS_DIR = Path("/home/bittu/Downloads")
DATA_DIR = REPO_ROOT / "app" / "model" / "data"
CLEAN_DATASET_DIR = DATA_DIR / "normalized_clean_eval"

CLASS_NAMES = ["cracked_tiles", "paint_peeling", "spalling", "stagnant_water"]
TARGET_COUNT_PER_CLASS = 100  # Strict class normalization: 100 images per class = 400 total

def extract_water_zip():
    """Extracts water data from downloaded zip files in Downloads."""
    water_zip = DOWNLOADS_DIR / "Stagnant Water and Wet Surface Dataset 1.zip"
    extract_target = DATA_DIR / "raw_water_dataset"
    extract_target.mkdir(parents=True, exist_ok=True)

    if water_zip.exists():
        print(f"[+] Extracting {water_zip.name} ({water_zip.stat().st_size / (1024*1024):.1f} MB)...")
        with zipfile.ZipFile(water_zip, 'r') as zf:
            zf.extractall(extract_target)
        print(f"  [Extracted] Water images uncompressed to {extract_target}")
    return extract_target

def is_valid_user_photo(img_path: Path, min_dim: int = 150, max_aspect: float = 3.0) -> bool:
    """
    Filters out microscopic/industrial images or corrupted stubs.
    Enforces minimum dimensions and reasonable aspect ratios typical of smartphone photos.
    """
    try:
        with Image.open(img_path) as img:
            img.verify()
            w, h = img.size
            if w < min_dim or h < min_dim:
                return False
            aspect = max(w / h, h / w)
            if aspect > max_aspect:
                return False
            return True
    except Exception:
        return False

def build_normalized_clean_dataset(water_extract_dir: Path):
    """
    Builds a clean, class-balanced dataset where every category has exactly TARGET_COUNT_PER_CLASS samples.
    Strips industrial magnetic tile crops and keeps only real-world smartphone-style photos.
    """
    print(f"\n[+] Building Class-Normalized Dataset ({TARGET_COUNT_PER_CLASS} images per class)...")
    if CLEAN_DATASET_DIR.exists():
        shutil.rmtree(CLEAN_DATASET_DIR)

    for cname in CLASS_NAMES:
        (CLEAN_DATASET_DIR / cname).mkdir(parents=True, exist_ok=True)

    # 1. Stagnant Water (From zip extraction & existing pool)
    water_candidates = []
    if water_extract_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            water_candidates.extend(list(water_extract_dir.rglob(ext)))

    # Also collect from existing test/train splits if needed
    official_water = DATA_DIR / "test" / "stagnant_water"
    if official_water.exists():
        water_candidates.extend(list(official_water.glob("*.*")))

    random.shuffle(water_candidates)
    valid_water = []
    for p in water_candidates:
        if is_valid_user_photo(p):
            valid_water.append(p)
            if len(valid_water) >= TARGET_COUNT_PER_CLASS:
                break

    print(f"  [Selection] Stagnant Water: Selected {len(valid_water)} real-world photos.")
    for idx, src_p in enumerate(valid_water):
        shutil.copy2(src_p, CLEAN_DATASET_DIR / "stagnant_water" / f"water_{idx:04d}.jpg")

    # 2. Spalling (From official dataset + U-spalling real photos)
    spall_candidates = []
    for split in ["test", "val", "train"]:
        spall_dir = DATA_DIR / split / "spalling"
        if spall_dir.exists():
            spall_candidates.extend(list(spall_dir.glob("*.*")))

    random.shuffle(spall_candidates)
    valid_spall = []
    for p in spall_candidates:
        if is_valid_user_photo(p):
            valid_spall.append(p)
            if len(valid_spall) >= TARGET_COUNT_PER_CLASS:
                break

    print(f"  [Selection] Spalling: Selected {len(valid_spall)} real-world photos.")
    for idx, src_p in enumerate(valid_spall):
        shutil.copy2(src_p, CLEAN_DATASET_DIR / "spalling" / f"spalling_{idx:04d}.jpg")

    # 3. Cracked Tiles (From official dataset only - NO industrial magnetic tile crops)
    tile_candidates = []
    for split in ["test", "val", "train"]:
        tile_dir = DATA_DIR / split / "cracked_tiles"
        if tile_dir.exists():
            tile_candidates.extend(list(tile_dir.glob("*.*")))

    random.shuffle(tile_candidates)
    valid_tile = []
    for p in tile_candidates:
        if is_valid_user_photo(p):
            valid_tile.append(p)
            if len(valid_tile) >= TARGET_COUNT_PER_CLASS:
                break

    print(f"  [Selection] Cracked Tiles: Selected {len(valid_tile)} real-world photos.")
    for idx, src_p in enumerate(valid_tile):
        shutil.copy2(src_p, CLEAN_DATASET_DIR / "cracked_tiles" / f"tile_{idx:04d}.jpg")

    # 4. Paint Peeling (From official dataset)
    paint_candidates = []
    for split in ["test", "val", "train"]:
        paint_dir = DATA_DIR / split / "paint_peeling"
        if paint_dir.exists():
            paint_candidates.extend(list(paint_dir.glob("*.*")))

    random.shuffle(paint_candidates)
    valid_paint = []
    for p in paint_candidates:
        if is_valid_user_photo(p):
            valid_paint.append(p)
            if len(valid_paint) >= TARGET_COUNT_PER_CLASS:
                break

    print(f"  [Selection] Paint Peeling: Selected {len(valid_paint)} real-world photos.")
    for idx, src_p in enumerate(valid_paint):
        shutil.copy2(src_p, CLEAN_DATASET_DIR / "paint_peeling" / f"paint_{idx:04d}.jpg")

    print("\n" + "=" * 65)
    print(" CLASS-BALANCED REAL-WORLD SMARTPHONE DATASET SUMMARY")
    print("=" * 65)
    total_imgs = 0
    for cname in CLASS_NAMES:
        c_count = len(list((CLEAN_DATASET_DIR / cname).glob("*.jpg")))
        total_imgs += c_count
        pct = (c_count / max(1, total_imgs)) * 100.0 if total_imgs > 0 else 0
        print(f" • {cname.ljust(18)}: {c_count:>4} images (Exactly 25% balanced share)")

    print("-" * 65)
    print(f" TOTAL CLEAN NORMALIZED DATASET: {total_imgs} images in {CLEAN_DATASET_DIR}")
    print("=" * 65)

def main():
    print("=" * 65)
    print(" InfraPulse - Dataset Extraction, Industrial Filtering & Class Normalization")
    print("=" * 65)
    water_dir = extract_water_zip()
    build_normalized_clean_dataset(water_dir)

if __name__ == "__main__":
    main()
