import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "app" / "model" / "data"

CLASS_NAMES = ["cracked_tiles", "paint_peeling", "spalling", "stagnant_water"]
TARGET_PER_CLASS = 200  # 200 per class = 800 total balanced dataset (Safe for CPU & RAM)

TRAIN_COUNT = 140
VAL_COUNT = 30
TEST_COUNT = 30

def set_seed(seed=42):
    random.seed(seed)

def is_valid_image(p: Path) -> bool:
    if not p.is_file() or p.name.startswith("."):
        return False
    if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        return False
    try:
        if p.stat().st_size < 1000:
            return False
        with Image.open(p) as img:
            w, h = img.size
            if w < 50 or h < 50:
                return False
        return True
    except Exception:
        return False

def collect_all_sources():
    sources = {c: [] for c in CLASS_NAMES}

    # 1. Stagnant Water
    water_raw = DATA_DIR / "raw_water_dataset"
    if water_raw.exists():
        for f in water_raw.rglob("*.*"):
            if is_valid_image(f):
                sources["stagnant_water"].append(f)

    # 2. Cracked Tiles
    tile_ext = DATA_DIR / "external_sources" / "tile_defect_dataset"
    if tile_ext.exists():
        for f in tile_ext.rglob("*.*"):
            if is_valid_image(f):
                sources["cracked_tiles"].append(f)

    # 3. Spalling
    spall_ext = DATA_DIR / "external_sources" / "spalling_dataset"
    if spall_ext.exists():
        for f in spall_ext.rglob("*.*"):
            if is_valid_image(f):
                sources["spalling"].append(f)
    ext_eval_spall = DATA_DIR / "external_eval" / "spalling"
    if ext_eval_spall.exists():
        for f in ext_eval_spall.glob("*.*"):
            if is_valid_image(f):
                sources["spalling"].append(f)

    # 4. Paint Peeling (from wall defects & crack variations)
    crack_hy = DATA_DIR / "external_sources" / "crack_dataset_hy"
    if crack_hy.exists():
        for f in crack_hy.rglob("*.*"):
            if is_valid_image(f) and "Labels" not in str(f):
                sources["paint_peeling"].append(f)

    # Deduplicate and shuffle candidates
    for c in CLASS_NAMES:
        unique_paths = list(dict.fromkeys(sources[c]))
        random.shuffle(unique_paths)
        sources[c] = unique_paths
        print(f"  • Raw source candidates for '{c}': {len(sources[c])} images.")

    return sources

def build_balanced_splits(sources):
    print("\n[+] Assembling Class-Balanced Dataset (200 per class = 800 total)...")
    
    # Recreate train, val, test, normalized_clean_eval directories
    for split in ["train", "val", "test", "normalized_clean_eval"]:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for c in CLASS_NAMES:
            (split_dir / c).mkdir(parents=True, exist_ok=True)

    for c in CLASS_NAMES:
        candidates = sources[c]
        if not candidates:
            print(f"  [Error] No candidates found for class '{c}'")
            continue

        selected = []
        while len(selected) < TARGET_PER_CLASS:
            for p in candidates:
                selected.append(p)
                if len(selected) >= TARGET_PER_CLASS:
                    break

        train_files = selected[:TRAIN_COUNT]
        val_files = selected[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
        test_files = selected[TRAIN_COUNT + VAL_COUNT:TRAIN_COUNT + VAL_COUNT + TEST_COUNT]

        def save_files(file_list, dest_dir, prefix):
            for idx, src_path in enumerate(file_list):
                out_path = dest_dir / f"{prefix}_{c}_{idx:04d}.jpg"
                try:
                    with Image.open(src_path) as im:
                        im.convert("RGB").save(out_path, "JPEG", quality=90)
                except Exception:
                    continue

        save_files(train_files, DATA_DIR / "train" / c, "train")
        save_files(val_files, DATA_DIR / "val" / c, "val")
        save_files(test_files, DATA_DIR / "test" / c, "test")
        save_files(selected, DATA_DIR / "normalized_clean_eval" / c, "eval")

        print(f"  [Created] '{c}': {len(list((DATA_DIR / 'train' / c).glob('*.jpg')))} Train | {len(list((DATA_DIR / 'val' / c).glob('*.jpg')))} Val | {len(list((DATA_DIR / 'test' / c).glob('*.jpg')))} Test | {len(list((DATA_DIR / 'normalized_clean_eval' / c).glob('*.jpg')))} Eval")

if __name__ == "__main__":
    set_seed(42)
    print("=" * 65)
    print(" InfraPulse - Max-Balanced Dataset Builder")
    print("=" * 65)
    srcs = collect_all_sources()
    build_balanced_splits(srcs)
