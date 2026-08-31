from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from app.config import BASE_DIR
from app.database import get_db
from app.models import Complaint, CategoryEnum, TicketComment, Notification
from app.schemas import ClassificationPayload, ComplaintResponse
from app.auth import get_current_user, get_current_staff, get_current_admin
from app.priority_queue import (
    compute_priority_score,
    get_category_live_queue,
    get_queue_position,
    mock_classify_defect
)

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

router = APIRouter(prefix="/api/v1", tags=["ML Model Integration & REST API"])

@router.post("/complaints/{complaint_id}/classify", response_model=ComplaintResponse)
async def classify_complaint_from_model(
    complaint_id: int,
    payload: ClassificationPayload,
    db: AsyncSession = Depends(get_db)
):
    complaint = await db.get(Complaint, complaint_id)
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Complaint ID {complaint_id} not found")
        
    try:
        category_enum = CategoryEnum(payload.category.value if hasattr(payload.category, 'value') else payload.category)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {payload.category}")

    complaint.defect_name = payload.defect_name.strip()
    complaint.category = category_enum
    complaint.severity = payload.severity
    complaint.extent = payload.extent
    complaint.priority_score = compute_priority_score(
        category_enum,
        payload.defect_name,
        payload.severity,
        payload.extent
    )
    
    await db.commit()
    await db.refresh(complaint)
    
    queue_pos = await get_queue_position(db, complaint)
    resp = ComplaintResponse.model_validate(complaint)
    resp.queue_position = queue_pos
    return resp

@router.post("/complaints/{complaint_id}/mock-classify")
async def trigger_mock_classification(
    complaint_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    complaint = await db.get(Complaint, complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
        
    mock_data = mock_classify_defect(complaint.description, complaint.photo_path)
    complaint.defect_name = mock_data["defect_name"]
    complaint.category = mock_data["category"]
    complaint.severity = mock_data["severity"]
    complaint.extent = mock_data["extent"]
    complaint.priority_score = compute_priority_score(
        mock_data["category"],
        mock_data["defect_name"],
        mock_data["severity"],
        mock_data["extent"]
    )
    
    await db.commit()
    await db.refresh(complaint)
    
    queue_pos = await get_queue_position(db, complaint)
    
    if "hx-request" in request.headers:
        return templates.TemplateResponse(request=request, name="components/complaint_card.html", context={
            "complaint": complaint,
            "queue_position": queue_pos
        })
        
    return {"message": "Classification updated", "complaint_id": complaint.id}

@router.get("/complaints/{complaint_id}", response_model=ComplaintResponse)
async def get_complaint_api(complaint_id: int, db: AsyncSession = Depends(get_db)):
    complaint = await db.get(Complaint, complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    queue_pos = await get_queue_position(db, complaint)
    resp = ComplaintResponse.model_validate(complaint)
    resp.queue_position = queue_pos
    return resp

@router.get("/queues/{category}", response_model=List[ComplaintResponse])
async def get_category_queue_api(category: str, db: AsyncSession = Depends(get_db)):
    try:
        cat_enum = CategoryEnum(category.capitalize())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid category")
        
    queue = await get_category_live_queue(db, cat_enum)
    result = []
    for idx, item in enumerate(queue, start=1):
        resp = ComplaintResponse.model_validate(item)
        resp.queue_position = idx
        result.append(resp)
    return result

@router.get("/tickets/{ticket_id}/comments")
async def get_ticket_comments_api(ticket_id: int, db: AsyncSession = Depends(get_db)):
    """
    Polling endpoint for live ticket comments & chat.
    """
    stmt = select(TicketComment).where(TicketComment.ticket_id == ticket_id).order_by(TicketComment.created_at.asc())
    res = await db.execute(stmt)
    comments = res.scalars().all()
    return [
        {
            "id": c.id,
            "sender_name": c.sender_name,
            "sender_role": c.sender_role,
            "message": c.message,
            "created_at": c.created_at.strftime('%b %d, %H:%M')
        }
        for c in comments
    ]

@router.get("/notifications")
async def get_notifications_api(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Returns live notifications for the logged-in user or staff.
    """
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    
    if not (user or staff or admin):
        return {"unread_count": 0, "notifications": []}

    stmt = select(Notification)
    if user:
        stmt = stmt.where(Notification.user_id == user.id)
    elif staff:
        stmt = stmt.where(Notification.staff_id == staff.id)

    stmt = stmt.order_by(Notification.created_at.desc()).limit(15)
    res = await db.execute(stmt)
    notifications = res.scalars().all()

    unread_count = sum(1 for n in notifications if not n.is_read)

    return {
        "unread_count": unread_count,
        "notifications": [
            {
                "id": n.id,
                "title": n.title,
                "message": n.message,
                "link_url": n.link_url,
                "is_read": n.is_read,
                "created_at": n.created_at.strftime('%b %d %H:%M')
            }
            for n in notifications
        ]
    }

@router.post("/notifications/mark-read")
async def mark_notifications_read(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)

    if user:
        stmt = select(Notification).where(Notification.user_id == user.id, Notification.is_read == False)
    elif staff:
        stmt = select(Notification).where(Notification.staff_id == staff.id, Notification.is_read == False)
    else:
        return {"status": "ok"}

    res = await db.execute(stmt)
    unread_notifs = res.scalars().all()
    for n in unread_notifs:
        n.is_read = True

    await db.commit()
    return {"status": "ok", "marked": len(unread_notifs)}
