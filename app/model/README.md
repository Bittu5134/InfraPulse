# InfraPulse V1

Baseline ML system for the IITK DesCon/Takneek InfraPulse problem.

## Architecture

EfficientNet-B0 pretrained on ImageNet only.

Classifier head:

Dropout -> Linear(1280,512) -> ReLU -> Dropout -> Linear(512,4)

Classes:

- cracked_tiles
- paint_peeling
- spalling
- stagnant_water

## Dataset structure

Create:

data/
  train/
    cracked_tiles/
    paint_peeling/
    spalling/
    stagnant_water/
  val/
    cracked_tiles/
    paint_peeling/
    spalling/
    stagnant_water/
  test/
    cracked_tiles/
    paint_peeling/
    spalling/
    stagnant_water/

With only ~116 images, avoid random image-level splitting if multiple
photographs come from the same physical defect/location. Split by scene/site
to reduce leakage.

## Install

python -m venv .venv

Windows:
.venv\Scripts\activate

Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

## Train

Run from project root:

python src/train.py \
  --data data \
  --phase1-epochs 5 \
  --epochs 25 \
  --image-size 224 \
  --batch-size 16 \
  --out checkpoints/best_infrapulse_v1.pt

Training has two phases:

1. Backbone frozen, head trained.
2. Last two EfficientNet feature blocks unfrozen.

Phase-2 learning rates:
- backbone: 1e-5
- classifier head: 1e-4

## Evaluate classification

python src/evaluate.py \
  --data data \
  --checkpoint checkpoints/best_infrapulse_v1.pt

## Run inference

python src/inference.py \
  --checkpoint checkpoints/best_infrapulse_v1.pt \
  --image example.jpg \
  --age-hours 4

During real testing, look for:

[INFO] ML inference pipeline loaded successfully
[INFO] MODEL_MODE = ML
[INFO] MODEL_LOADED = True
[INFO] FALLBACK_USED = False

For evaluation/debugging, strongly consider using `--no-fallback` so a broken
ML pipeline fails visibly instead of hiding behind the heuristic:

python src/inference.py \
  --checkpoint checkpoints/best_infrapulse_v1.pt \
  --image example.jpg \
  --no-fallback

## Priority score

PriorityScore =
    TypeTier*1000
    + Severity*5
    + Extent*3
    + TimeBonus

TypeTier:
- cracked_tiles: 1
- paint_peeling: 0
- other queues: 0

TimeBonus = age_hours * 0.1

Severity and extent are heuristic proxies calculated from GradCAM++.

Important: GradCAM++ is not true segmentation. Version 2 should replace
GradCAM-derived extent with an actual segmentation head if the baseline
priority ranking is weak.

## Important competition-rule note

Use only generic ImageNet pretrained weights for EfficientNet-B0.
Do not load any model/checkpoint pretrained specifically for:
- defects
- building damage
- stagnant water
- cracks/spalling/maintenance damage

Document all external libraries and datasets used.
