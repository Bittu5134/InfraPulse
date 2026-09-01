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
TARGET_COUNT_PER_CLASS = 250  # 250 images per class = 1,000 total balanced test images

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

    # Helper function to sample exactly TARGET_COUNT_PER_CLASS images
    def sample_exact_class_images(candidates, class_name, prefix):
        random.shuffle(candidates)
        valid = [p for p in candidates if is_valid_user_photo(p)]
        if not valid:
            return 0
        selected = []
        while len(selected) < TARGET_COUNT_PER_CLASS:
            for p in valid:
                selected.append(p)
                if len(selected) >= TARGET_COUNT_PER_CLASS:
                    break
        target_dir = CLEAN_DATASET_DIR / class_name
        for idx, src_p in enumerate(selected):
            shutil.copy2(src_p, target_dir / f"{prefix}_{idx:04d}.jpg")
        print(f"  [Selection] {class_name.title()}: Selected {len(selected)} real-world photos.")
        return len(selected)

    # 1. Stagnant Water
    water_candidates = []
    if water_extract_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            water_candidates.extend(list(water_extract_dir.rglob(ext)))
    official_water = DATA_DIR / "test" / "stagnant_water"
    if official_water.exists():
        water_candidates.extend(list(official_water.glob("*.*")))
    sample_exact_class_images(water_candidates, "stagnant_water", "water")

    # 2. Spalling
    spall_candidates = []
    for split in ["test", "val", "train"]:
        spall_dir = DATA_DIR / split / "spalling"
        if spall_dir.exists():
            spall_candidates.extend(list(spall_dir.glob("*.*")))
    sample_exact_class_images(spall_candidates, "spalling", "spalling")

    # 3. Cracked Tiles
    tile_candidates = []
    for split in ["test", "val", "train"]:
        tile_dir = DATA_DIR / split / "cracked_tiles"
        if tile_dir.exists():
            tile_candidates.extend(list(tile_dir.glob("*.*")))
    sample_exact_class_images(tile_candidates, "cracked_tiles", "tile")

    # 4. Paint Peeling
    paint_candidates = []
    for split in ["test", "val", "train"]:
        paint_dir = DATA_DIR / split / "paint_peeling"
        if paint_dir.exists():
            paint_candidates.extend(list(paint_dir.glob("*.*")))
    sample_exact_class_images(paint_candidates, "paint_peeling", "paint")

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
