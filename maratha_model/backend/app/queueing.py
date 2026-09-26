from sqlalchemy.orm import Session

from .models import Complaint, Category, Status


def live_queue(db: Session, category: Category) -> list[Complaint]:
    """Complaints for a category, excluding Resolved, ranked by
    (defect-type priority, severity score) descending — most urgent first."""
    complaints = (
        db.query(Complaint)
        .filter(Complaint.category == category, Complaint.status != Status.resolved)
        .all()
    )
    complaints.sort(key=lambda c: (c.type_priority, c.severity_score), reverse=True)
    return complaints


def with_positions(complaints: list[Complaint]) -> list[dict]:
    out = []
    for i, c in enumerate(complaints):
        out.append({
            "id": c.id,
            "name": c.name,
            "address": c.address,
            "description": c.description,
            "image_path": c.image_path,
            "defect_label": c.defect_label,
            "category": c.category,
            "confidence": c.confidence,
            "severity_score": c.severity_score,
            "type_priority": c.type_priority,
            "status": c.status,
            "created_at": c.created_at,
            "queue_position": i + 1,
        })
    return out
