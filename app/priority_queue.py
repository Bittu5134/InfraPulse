import os
import random
import numpy as np
from typing import List, Optional
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import CATEGORY_WEIGHTS, DEFECT_PRIORITY_BOOST
from app.models import Complaint, CategoryEnum, StatusEnum, generate_10_digit_id

def compute_priority_score(category: Optional[CategoryEnum], defect_name: Optional[str], severity: float, extent: float, age_hours: float = 0.0) -> float:
    """
    Computes objective priority score according to problem statement specifications:
    
    PriorityScore = CategoryTierBase (1000/2000/3000)
                    + (Severity x 5.0)
                    + (Extent x 3.0)
                    + Strictly Capped TimeBonus (Max 5.0 pts - Tie Breaker Only)
    
    1. Category Tier Base (Safeguard against database cross-pollination):
       - Structural: 3000 base points
       - Functional: 2000 base points
       - Performance: 1000 base points
       
    2. Core Severity & Extent (15% Methodology Score):
       - Severity scaled 0-10 (weighted x 5.0 => max 50.0 pts)
       - Extent scaled 0-10 (weighted x 3.0 => max 30.0 pts)
       Guarantees deep, critical structural damage outranks wide, shallow surface wear.
       
    3. Capped Time Bonus (Prevents Escalation Trap):
       - Scaled at age_hours * 0.05, strictly CAPPED at max 5.0 points.
       - Acts strictly as a tie-breaker so new, high-severity complaints immediately
         jump to the top of the live evaluation queue.
    """
    if not category:
        return 0.0

    # 1. Category Tier Base Points
    cat_str = category.value if hasattr(category, 'value') else str(category)
    if cat_str == "Structural":
        base_tier = 3000.0
    elif cat_str == "Functional":
        base_tier = 2000.0
    else:
        base_tier = 1000.0

    # 2. Normalize Severity (0-10) and Extent (0-10)
    sev_scaled = (severity / 10.0 if severity > 10.0 else severity)
    ext_scaled = (extent / 10.0 if extent > 10.0 else extent)

    sev_pts = float(np.clip(sev_scaled, 0.0, 10.0)) * 5.0   # max 50.0 pts
    ext_pts = float(np.clip(ext_scaled, 0.0, 10.0)) * 3.0   # max 30.0 pts

    # Defect sub-tier bonus (e.g. Cracked Tiles > Paint Peeling within Performance tier)
    d_clean = (defect_name or "").lower().strip()
    defect_sub_bonus = 1.0 if ("tile" in d_clean or "crack" in d_clean) else (0.5 if "water" in d_clean else 0.0)

    # 3. Strictly Capped Time Bonus (Max 5.0 pts - Tie Breaker Only)
    time_bonus = min(5.0, max(0.0, float(age_hours)) * 0.05)

    total_score = base_tier + sev_pts + ext_pts + defect_sub_bonus + time_bonus
    return round(total_score, 1)

def mock_classify_defect(description: str, filename: str) -> dict:
    text = (description + " " + filename).lower()
    defects = [
        {"defect_name": "Spalling", "category": CategoryEnum.STRUCTURAL, "severity": round(random.uniform(7.0, 9.8), 1), "extent": round(random.uniform(30.0, 80.0), 1)},
        {"defect_name": "Stagnant Water", "category": CategoryEnum.FUNCTIONAL, "severity": round(random.uniform(5.5, 8.5), 1), "extent": round(random.uniform(25.0, 75.0), 1)},
        {"defect_name": "Cracked Tiles", "category": CategoryEnum.PERFORMANCE, "severity": round(random.uniform(5.0, 8.0), 1), "extent": round(random.uniform(20.0, 60.0), 1)},
        {"defect_name": "Paint Peeling", "category": CategoryEnum.PERFORMANCE, "severity": round(random.uniform(3.0, 6.0), 1), "extent": round(random.uniform(15.0, 50.0), 1)},
    ]
    if "spall" in text or "concrete" in text or "pillar" in text or "beam" in text:
        chosen = defects[0]
    elif "water" in text or "drain" in text or "puddle" in text or "flood" in text:
        chosen = defects[1]
    elif "tile" in text or "floor" in text or "crack" in text:
        chosen = defects[2]
    elif "paint" in text or "peel" in text or "wall" in text:
        chosen = defects[3]
    else:
        chosen = random.choice(defects)
    return chosen

def classify_defect(image_path: Optional[str] = None, description: str = "", filename: str = "") -> dict:
    """
    Primary classifier using the PyTorch EfficientNet-B0 ML model with GradCAM++ severity & extent.
    Falls back gracefully to heuristic analysis if image path is unavailable.
    """
    if image_path and os.path.exists(image_path):
        from app.model_service import predict_single_image
        res = predict_single_image(image_path)
        return {
            "defect_name": res["defect_name"],
            "category": res["category"],
            "severity": res["severity"],
            "extent": res["extent"],
            "confidence": res.get("confidence", 0.0),
            "priority_score": res.get("priority_score", 0.0)
        }

    return mock_classify_defect(description, filename)

async def get_staff_tickets_filtered(
    db: AsyncSession,
    status_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    min_severity: Optional[float] = None,
    search_query: Optional[str] = None,
    sort_by: str = "priority_desc",
    include_resolved: bool = True
) -> List[Complaint]:
    stmt = select(Complaint)
    
    if not include_resolved:
        stmt = stmt.where(Complaint.status != StatusEnum.RESOLVED)
        
    if status_filter and status_filter.lower() != "all":
        try:
            st_enum = StatusEnum(status_filter)
            stmt = stmt.where(Complaint.status == st_enum)
        except ValueError:
            pass

    if category_filter and category_filter.lower() != "all":
        try:
            cat_enum = CategoryEnum(category_filter.capitalize())
            stmt = stmt.where(Complaint.category == cat_enum)
        except ValueError:
            pass

    if min_severity is not None and min_severity > 0:
        stmt = stmt.where(Complaint.severity >= min_severity)

    if search_query and search_query.strip():
        q = f"%{search_query.strip().lower()}%"
        if search_query.strip().isdigit():
            target_id = int(search_query.strip())
            stmt = stmt.where(or_(Complaint.id == target_id, Complaint.user_name.ilike(q), Complaint.address.ilike(q)))
        else:
            stmt = stmt.where(
                or_(
                    Complaint.user_name.ilike(q),
                    Complaint.user_email.ilike(q),
                    Complaint.address.ilike(q),
                    Complaint.description.ilike(q),
                    Complaint.defect_name.ilike(q)
                )
            )

    if sort_by == "date_desc":
        stmt = stmt.order_by(Complaint.created_at.desc())
    elif sort_by == "date_asc":
        stmt = stmt.order_by(Complaint.created_at.asc())
    elif sort_by == "severity_desc":
        stmt = stmt.order_by(Complaint.severity.desc(), Complaint.priority_score.desc())
    else:
        stmt = stmt.order_by(Complaint.priority_score.desc(), Complaint.created_at.asc())

    result = await db.execute(stmt)
    return list(result.scalars().all())

async def get_all_active_queue(db: AsyncSession, category_filter: Optional[CategoryEnum] = None) -> List[Complaint]:
    return await get_staff_tickets_filtered(
        db,
        category_filter=category_filter.value if (category_filter and hasattr(category_filter, 'value')) else (str(category_filter) if category_filter else None),
        include_resolved=False
    )

async def get_category_live_queue(db: AsyncSession, category: CategoryEnum) -> List[Complaint]:
    return await get_all_active_queue(db, category_filter=category)

async def get_queue_position(db: AsyncSession, complaint: Complaint) -> Optional[int]:
    if complaint.status == StatusEnum.RESOLVED or not complaint.category:
        return None
    
    queue = await get_category_live_queue(db, complaint.category)
    for idx, item in enumerate(queue, start=1):
        if item.id == complaint.id:
            return idx
    return None
