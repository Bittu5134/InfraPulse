import random
from typing import List, Optional
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import CATEGORY_WEIGHTS, DEFECT_PRIORITY_BOOST
from app.models import Complaint, CategoryEnum, StatusEnum, generate_10_digit_id

def compute_priority_score(category: Optional[CategoryEnum], defect_name: Optional[str], severity: float, extent: float) -> float:
    if not category:
        return 0.0
    
    cat_weight = CATEGORY_WEIGHTS.get(category.value, 1.0) if hasattr(category, 'value') else CATEGORY_WEIGHTS.get(str(category), 1.0)
    
    d_name_lower = (defect_name or "").lower().strip()
    defect_boost = 1.0
    for key, weight in DEFECT_PRIORITY_BOOST.items():
        if key in d_name_lower:
            defect_boost = weight
            break
            
    severity_component = severity * 0.6
    extent_component = (extent / 100.0) * 4.0
    
    total_score = (severity_component + extent_component + defect_boost) * cat_weight
    return round(total_score, 2)

def mock_classify_defect(description: str, filename: str) -> dict:
    """
    Generates realistic randomized mockup ML model output for defect classification.
    """
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
        
    return {
        "defect_name": chosen["defect_name"],
        "category": chosen["category"],
        "severity": chosen["severity"],
        "extent": chosen["extent"]
    }

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
    
    queue = await get_all_active_queue(db)
    for idx, item in enumerate(queue, start=1):
        if item.id == complaint.id:
            return idx
    return None
