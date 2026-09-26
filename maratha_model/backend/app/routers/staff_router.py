from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import schemas
from ..auth import require_staff
from ..database import get_db
from ..models import Account, Complaint, Status
from ..queueing import live_queue, with_positions
from ..routers.complaints_router import _jsonable
from ..ws import manager

router = APIRouter(prefix="/staff", tags=["staff"])


@router.get("/queue", response_model=list[schemas.ComplaintOut])
def get_queue(db: Session = Depends(get_db), staff: Account = Depends(require_staff)):
    queue = with_positions(live_queue(db, staff.staff_category))
    return [
        schemas.ComplaintOut(
            id=q["id"], name=q["name"], address=q["address"], description=q["description"],
            image_path=q["image_path"], defect_label=q["defect_label"], category=q["category"],
            confidence=q["confidence"], severity_score=q["severity_score"],
            type_priority=q["type_priority"], status=q["status"], created_at=q["created_at"],
            queue_position=q["queue_position"],
        )
        for q in queue
    ]


@router.patch("/complaints/{complaint_id}/status", response_model=schemas.ComplaintOut)
async def update_status(
    complaint_id: str,
    payload: schemas.StatusUpdate,
    db: Session = Depends(get_db),
    staff: Account = Depends(require_staff),
):
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(404, "Complaint not found")
    if complaint.category != staff.staff_category:
        raise HTTPException(403, "Not your category")

    complaint.status = payload.status
    db.commit()
    db.refresh(complaint)

    queue = with_positions(live_queue(db, complaint.category))
    await manager.broadcast(
        "queue_update", {"category": complaint.category.value, "queue": _jsonable(queue)}
    )

    position = None
    if complaint.status != Status.resolved:
        position = next((q["queue_position"] for q in queue if q["id"] == complaint.id), None)

    return schemas.ComplaintOut(
        id=complaint.id, name=complaint.name, address=complaint.address,
        description=complaint.description, image_path=complaint.image_path,
        defect_label=complaint.defect_label, category=complaint.category,
        confidence=complaint.confidence, severity_score=complaint.severity_score,
        type_priority=complaint.type_priority, status=complaint.status,
        created_at=complaint.created_at, queue_position=position,
    )
