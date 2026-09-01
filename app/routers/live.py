from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import BASE_DIR
from app.database import get_db
from app.models import Complaint, CategoryEnum
from app.priority_queue import get_category_live_queue, get_queue_position

from app.templates_config import templates

router = APIRouter(prefix="/live", tags=["HTMX Live Sync"])

@router.get("/queue/{category_str}", response_class=HTMLResponse)
async def live_queue_partial(category_str: str, request: Request, db: AsyncSession = Depends(get_db)):
    try:
        category = CategoryEnum(category_str.capitalize())
    except ValueError:
        raise HTTPException(status_code=404, detail="Category not found")
        
    complaints = await get_category_live_queue(db, category)
    return templates.TemplateResponse(request=request, name="components/queue_table.html", context={
        "category": category.value,
        "complaints": complaints
    })

@router.get("/complaint/{complaint_id}", response_class=HTMLResponse)
async def live_complaint_partial(complaint_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    complaint = await db.get(Complaint, complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
        
    queue_pos = await get_queue_position(db, complaint)
    return templates.TemplateResponse(request=request, name="components/complaint_card.html", context={
        "complaint": complaint,
        "queue_position": queue_pos
    })
