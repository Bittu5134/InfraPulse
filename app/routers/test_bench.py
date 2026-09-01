import os
import math
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Request, Query, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import BASE_DIR
from app.database import get_db
from app.auth import get_current_user, get_current_staff, get_current_admin
from app.model_service import predict_single_image, run_rule_based_classifier

from app.templates_config import templates

router = APIRouter(tags=["Model Benchmark"])

DATA_DIR = BASE_DIR / "app" / "model" / "data"

GROUND_TRUTH_CATEGORY_MAP = {
    "spalling": "Structural",
    "stagnant_water": "Functional",
    "cracked_tiles": "Performance",
    "paint_peeling": "Performance",
}

def scan_dataset_images(split: str = "test", class_filter: str = "all") -> List[Dict[str, str]]:
    images = []
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        return images

    classes = ["cracked_tiles", "paint_peeling", "spalling", "stagnant_water"]
    if class_filter != "all" and class_filter in classes:
        classes = [class_filter]

    for cls in classes:
        cls_dir = split_dir / cls
        if cls_dir.exists():
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    full_path = cls_dir / fname
                    rel_url = f"/test_data/{split}/{cls}/{fname}"
                    images.append({
                        "filename": fname,
                        "ground_truth_class": cls,
                        "ground_truth_name": cls.replace("_", " ").title(),
                        "ground_truth_category": GROUND_TRUTH_CATEGORY_MAP.get(cls, "Unknown"),
                        "full_path": str(full_path),
                        "img_url": rel_url,
                        "split": split
                    })
    return images

@router.get("/test", response_class=HTMLResponse)
async def render_benchmark_page(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    split: str = Query("test"),
    class_filter: str = Query("all"),
    db: AsyncSession = Depends(get_db)
):
    current_user = await get_current_user(request, db)
    current_staff = await get_current_staff(request, db)
    current_admin = await get_current_admin(request, db)

    # 1. Scan images from requested split & class filter
    all_images = scan_dataset_images(split=split, class_filter=class_filter)
    total_images = len(all_images)
    total_pages = max(1, math.ceil(total_images / page_size))
    current_page = min(page, total_pages)

    # 2. Slice to current page only (Lazy Evaluation / Resource Protection)
    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    page_images = all_images[start_idx:end_idx]

    # 3. Run predictions on this page's batch only
    items = []
    for img in page_images:
        # Deep Learning PyTorch Model Prediction
        ml_res = predict_single_image(img["full_path"])
        
        # Rule-Based / Mock Baseline Prediction
        rule_res = run_rule_based_classifier(img["filename"], "")

        # Check correctness against ground truth
        predicted_cls_slug = ml_res["defect_name"].lower().replace(" ", "_")
        is_ml_correct = (predicted_cls_slug == img["ground_truth_class"])
        is_rule_correct = (rule_res["defect_name"].lower().replace(" ", "_") == img["ground_truth_class"])

        items.append({
            "meta": img,
            "ml_model": ml_res,
            "rule_model": rule_res,
            "is_ml_correct": is_ml_correct,
            "is_rule_correct": is_rule_correct
        })

    return templates.TemplateResponse(
        request=request,
        name="test_bench.html",
        context={
            "current_user": current_user,
            "current_staff": current_staff,
            "current_admin": current_admin,
            "items": items,
            "current_page": current_page,
            "total_pages": total_pages,
            "total_images": total_images,
            "page_size": page_size,
            "split": split,
            "class_filter": class_filter,
            "classes": ["cracked_tiles", "paint_peeling", "spalling", "stagnant_water"]
        }
    )
