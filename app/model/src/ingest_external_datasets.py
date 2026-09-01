import os
import shutil
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXT_SOURCES = REPO_ROOT / "app" / "model" / "data" / "external_sources"
TARGET_DIR = REPO_ROOT / "app" / "model" / "data" / "external_eval"

CATEGORIES = ["spalling", "stagnant_water", "cracked_tiles", "paint_peeling"]

def initialize_directories():
    for cat in CATEGORIES:
        (TARGET_DIR / cat).mkdir(parents=True, exist_ok=True)

def ingest_spalling_images():
    """Ingests concrete spalling images from U-spalling-dataset."""
    spall_source = EXT_SOURCES / "spalling_dataset"
    target_spall = TARGET_DIR / "spalling"
    count = 0

    if spall_source.exists():
        imgs = [f for f in spall_source.rglob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        for i, img_path in enumerate(imgs):
            try:
                out_name = f"ext_spalling_{i:04d}.jpg"
                with Image.open(img_path) as im:
                    im.convert("RGB").save(target_spall / out_name, "JPEG")
                count += 1
            except Exception:
                continue
    print(f"  [Ingested] Spalling: {count} images copied to {target_spall}")
    return count

def ingest_tile_images():
    """Ingests tile defect images from Magnetic-tile-defect-datasets."""
    tile_source = EXT_SOURCES / "tile_defect_dataset"
    target_tile = TARGET_DIR / "cracked_tiles"
    count = 0

    if tile_source.exists():
        imgs = list(tile_source.rglob("*.jpg")) + list(tile_source.rglob("*.png")) + list(tile_source.rglob("*.bmp"))
        for i, img_path in enumerate(imgs[:500]):  # Sample 500 clean tile defect images
            try:
                # Convert BMP/PNG to clean JPG
                out_name = f"ext_tile_{i:04d}.jpg"
                with Image.open(img_path) as im:
                    im.convert("RGB").save(target_tile / out_name, "JPEG")
                count += 1
            except Exception:
                continue
    print(f"  [Ingested] Cracked Tiles: {count} images processed and copied to {target_tile}")
    return count

def ingest_puddle_images():
    """Ingests puddle/stagnant water images."""
    target_puddle = TARGET_DIR / "stagnant_water"
    count = 0

    puddle_sources = [EXT_SOURCES / "puddle_dataset", EXT_SOURCES / "water_puddle_paper"]
    for p_src in puddle_sources:
        if p_src.exists():
            imgs = list(p_src.rglob("*.jpg")) + list(p_src.rglob("*.png"))
            for img_path in imgs:
                try:
                    out_name = f"ext_puddle_{count:04d}.jpg"
                    with Image.open(img_path) as im:
                        im.convert("RGB").save(target_puddle / out_name, "JPEG")
                    count += 1
                except Exception:
                    continue
    print(f"  [Ingested] Stagnant Water: {count} images copied to {target_puddle}")
    return count

def ingest_wall_paint_images():
    """Ingests wall paint peeling & surface defect images."""
    target_paint = TARGET_DIR / "paint_peeling"
    wall_source = EXT_SOURCES / "building_wall_defects"
    count = 0

    if wall_source.exists():
        imgs = list(wall_source.rglob("*.jpg")) + list(wall_source.rglob("*.png"))
        for img_path in imgs:
            try:
                out_name = f"ext_paint_{count:04d}.jpg"
                with Image.open(img_path) as im:
                    im.convert("RGB").save(target_paint / out_name, "JPEG")
                count += 1
            except Exception:
                continue
    print(f"  [Ingested] Paint Peeling: {count} images copied to {target_paint}")
    return count

def main():
    print("=" * 65)
    print(" Ingesting Downloaded External Defect Datasets")
    print("=" * 65)
    initialize_directories()

    c_spall = ingest_spalling_images()
    c_tile = ingest_tile_images()
    c_water = ingest_puddle_images()
    c_paint = ingest_wall_paint_images()

    total = c_spall + c_tile + c_water + c_paint
    print("-" * 65)
    print(f"[+] Total External Dataset Samples Ingested: {total} images in {TARGET_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    main()
