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
from app.model_service import predict_all_models, get_models_leaderboard

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

    # 3. Multi-Model Benchmark Inference on this batch
    items = []
    for img in page_images:
        multi_eval = predict_all_models(
            image_path=img["full_path"],
            ground_truth_name=img["ground_truth_name"]
        )

        items.append({
            "meta": img,
            "models": multi_eval["models"],
            "winner": multi_eval["winner"]
        })

    leaderboard = get_models_leaderboard()

    return templates.TemplateResponse(
        request=request,
        name="test_bench.html",
        context={
            "current_user": current_user,
            "current_staff": current_staff,
            "current_admin": current_admin,
            "items": items,
            "leaderboard": leaderboard,
            "current_page": current_page,
            "total_pages": total_pages,
            "total_images": total_images,
            "page_size": page_size,
            "split": split,
            "class_filter": class_filter,
            "classes": ["cracked_tiles", "paint_peeling", "spalling", "stagnant_water"]
        }
    )

@router.get("/test/playground", response_class=HTMLResponse)
async def render_test_playground(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Renders an interactive custom playground where users can upload any photo,
    write custom descriptions, and receive live side-by-side outputs across all models.
    """
    current_user = await get_current_user(request, db)
    current_staff = await get_current_staff(request, db)
    current_admin = await get_current_admin(request, db)
    leaderboard = get_models_leaderboard()

    return templates.TemplateResponse(
        request=request,
        name="playground.html",
        context={
            "current_user": current_user,
            "current_staff": current_staff,
            "current_admin": current_admin,
            "leaderboard": leaderboard,
            "result": None,
        }
    )

@router.post("/test/playground", response_class=HTMLResponse)
@router.post("/api/test/analyze-custom")
async def analyze_custom_playground(
    request: Request,
    photo: Optional[Any] = None,
    description: str = "",
    db: AsyncSession = Depends(get_db)
):
    """
    Analyzes an uploaded photo and description across all models in parallel,
    returning structured outputs and highlighting the Clear Winner.
    """
    import io
    import uuid
    from PIL import Image
    from fastapi import UploadFile, Form, File
    from fastapi.responses import JSONResponse

    form = await request.form()
    uploaded_file = form.get("photo")
    desc = form.get("description", "") or ""

    if not uploaded_file or not hasattr(uploaded_file, "read"):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "No image file provided."}, status_code=400)
        return templates.TemplateResponse(
            request=request,
            name="playground.html",
            context={
                "current_user": await get_current_user(request, db),
                "current_staff": await get_current_staff(request, db),
                "current_admin": await get_current_admin(request, db),
                "leaderboard": get_models_leaderboard(),
                "error": "Please select or drop an image file to analyze.",
                "result": None,
            }
        )

    try:
        content = await uploaded_file.read()
        pil_img = Image.open(io.BytesIO(content))
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGBA")
        else:
            pil_img = pil_img.convert("RGB")

        # Save to uploads directory
        filename = f"custom_test_{uuid.uuid4().hex[:10]}.png"
        upload_dir = BASE_DIR / "app" / "static" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved_path = upload_dir / filename
        pil_img.save(saved_path, format="PNG")
        image_url = f"/static/uploads/{filename}"

        # Evaluate across all models
        evaluation = predict_all_models(image_path=str(saved_path), description=desc)

        result_data = {
            "image_url": image_url,
            "filename": uploaded_file.filename or filename,
            "description": desc,
            "models": evaluation["models"],
            "winner": evaluation["winner"],
        }

        # If AJAX request or API endpoint, return JSON
        if request.headers.get("accept") == "application/json" or request.url.path.startswith("/api/"):
            return JSONResponse(result_data)

        # Render HTML with results
        return templates.TemplateResponse(
            request=request,
            name="playground.html",
            context={
                "current_user": await get_current_user(request, db),
                "current_staff": await get_current_staff(request, db),
                "current_admin": await get_current_admin(request, db),
                "leaderboard": get_models_leaderboard(),
                "result": result_data,
            }
        )
    except Exception as e:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": str(e)}, status_code=500)
        return templates.TemplateResponse(
            request=request,
            name="playground.html",
            context={
                "current_user": await get_current_user(request, db),
                "current_staff": await get_current_staff(request, db),
                "current_admin": await get_current_admin(request, db),
                "leaderboard": get_models_leaderboard(),
                "error": f"Error analyzing photo: {str(e)}",
                "result": None,
            }
        )

