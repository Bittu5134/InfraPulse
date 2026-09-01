"""
pull_bd3_dataset.py

Downloads the full BD3 building-defect dataset (chandrabhuma/building_defect_vqa
on Hugging Face) and merges it into InfraPulse's data/train|val|test/<class>/
folder structure, alongside your own ~116 site-split photos.

Run this from your project root (the same folder that contains train.py,
dataset.py, model.py) so the output paths line up with what train.py expects.

Setup (one-time):
    pip install datasets pillow scikit-learn

Then:
    python pull_bd3_dataset.py
"""

import random
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ---- 1. config ----

OUT_ROOT = Path("data")  # matches data/train, data/val, data/test used by train.py
SEED = 42
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# Map BD3's labels onto your 4 classes. Anything not listed here is skipped
# (algae, stain, plain/normal are not part of your taxonomy).
LABEL_MAP = {
    "peeling": "paint_peeling",
    "spalling": "spalling",
    "major_crack": "cracked_tiles",
    "minor_crack": "cracked_tiles",
}

# Soft cap per class so cracked_tiles (which draws from two source classes,
# ~1200 images combined) doesn't dwarf spalling (only ~500 images total).
MAX_PER_CLASS = 550

random.seed(SEED)


def main():
    print("Downloading BD3 dataset from Hugging Face (~795 MB, one-time)...")
    ds = load_dataset("chandrabhuma/building_defect_vqa")

    # ---- 2. filter + remap + bucket by target class ----
    buckets = {cls: [] for cls in set(LABEL_MAP.values())}

    for split_name in ds.keys():  # "train" and "test" splits inside the HF dataset
        for row in ds[split_name]:
            target_class = LABEL_MAP.get(row["answer"])
            if target_class is None:
                continue
            buckets[target_class].append(row["image"])

    for cls, imgs in buckets.items():
        print(f"{cls}: {len(imgs)} images available before capping")

    # ---- 3. cap, shuffle, split 70/15/15, save as jpg ----
    for cls, imgs in buckets.items():
        random.shuffle(imgs)
        imgs = imgs[:MAX_PER_CLASS]

        train_imgs, temp_imgs = train_test_split(
            imgs, train_size=SPLIT["train"], random_state=SEED
        )
        val_frac = SPLIT["val"] / (SPLIT["val"] + SPLIT["test"])
        val_imgs, test_imgs = train_test_split(
            temp_imgs, train_size=val_frac, random_state=SEED
        )

        for subset_name, subset_imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            out_dir = OUT_ROOT / subset_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(subset_imgs):
                img.convert("RGB").save(out_dir / f"bd3_{cls}_{i:04d}.jpg", quality=90)

        print(
            f"{cls}: wrote {len(train_imgs)} train / "
            f"{len(val_imgs)} val / {len(test_imgs)} test"
        )

    print(
        "\nDone. Your own ~116 site-split photos are untouched — "
        "this only added BD3 images into the same class folders."
    )


if __name__ == "__main__":
    main()
