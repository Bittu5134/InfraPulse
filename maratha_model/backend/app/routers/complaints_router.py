import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from .. import schemas
from ..auth import get_current_account, require_user
from ..database import get_db
from ..ml.classifier import get_classifier
from ..ml.severity import compute_severity
from ..models import Account, Complaint, DEFECT_TO_CATEGORY, DEFECT_TYPE_PRIORITY, Status
from ..queueing import live_queue, with_positions
from ..ws import manager

router = APIRouter(prefix="/complaints", tags=["complaints"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", response_model=schemas.ComplaintOut)
async def submit_complaint(
    name: str = Form(...),
    address: str = Form(...),
    description: str = Form(""),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
    account: Account = Depends(require_user),
):
    ext = Path(photo.filename).suffix or ".jpg"
    fname = f"{uuid.uuid4()}{ext}"
    fpath = UPLOAD_DIR / fname
    contents = await photo.read()
    fpath.write_bytes(contents)

    defect_label, confidence = get_classifier().predict(str(fpath))
    severity = compute_severity(str(fpath), defect_label)
    category = DEFECT_TO_CATEGORY[defect_label]
    type_priority = DEFECT_TYPE_PRIORITY[defect_label]

    complaint = Complaint(
        owner_id=account.id,
        name=name,
        address=address,
        description=description,
        image_path=f"/uploads/{fname}",
        defect_label=defect_label,
        category=category,
        confidence=confidence,
        severity_score=severity,
        type_priority=type_priority,
        status=Status.submitted,
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)

    queue = with_positions(live_queue(db, category))
    await manager.broadcast("queue_update", {"category": category.value, "queue": _jsonable(queue)})

    position = next((q["queue_position"] for q in queue if q["id"] == complaint.id), None)
    return _to_out(complaint, position)


@router.get("/mine", response_model=list[schemas.ComplaintOut])
def my_complaints(db: Session = Depends(get_db), account: Account = Depends(require_user)):
    complaints = db.query(Complaint).filter(Complaint.owner_id == account.id).all()
    results = []
    for c in complaints:
        position = None
        if c.status != Status.resolved:
            queue = live_queue(db, c.category)
            position = next((i + 1 for i, q in enumerate(queue) if q.id == c.id), None)
        results.append(_to_out(c, position))
    return results


def _to_out(c: Complaint, position):
    return schemas.ComplaintOut(
        id=c.id, name=c.name, address=c.address, description=c.description,
        image_path=c.image_path, defect_label=c.defect_label, category=c.category,
        confidence=c.confidence, severity_score=c.severity_score, type_priority=c.type_priority,
        status=c.status, created_at=c.created_at, queue_position=position,
    )


def _jsonable(queue: list[dict]) -> list[dict]:
    out = []
    for q in queue:
        item = dict(q)
        item["category"] = item["category"].value if hasattr(item["category"], "value") else item["category"]
        item["status"] = item["status"].value if hasattr(item["status"], "value") else item["status"]
        item["created_at"] = item["created_at"].isoformat()
        out.append(item)
    return out
