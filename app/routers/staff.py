from typing import Optional
from fastapi import APIRouter, Depends, Request, Form, HTTPException, Query, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import BASE_DIR
from app.database import get_db
from app.models import Staff, Complaint, CategoryEnum, StatusEnum
from app.auth import verify_password, get_current_staff, get_current_user, require_staff, get_current_admin
from app.priority_queue import get_all_active_queue

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

router = APIRouter(prefix="/staff", tags=["Staff Portal"])

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def staff_index(request: Request, db: AsyncSession = Depends(get_db)):
    staff = await get_current_staff(request, db)
    if staff:
        return RedirectResponse(url="/staff/queue", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url="/staff/login", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/login", response_class=HTMLResponse)
async def staff_login_page(request: Request, db: AsyncSession = Depends(get_db)):
    staff = await get_current_staff(request, db)
    user = await get_current_user(request, db)
    admin = await get_current_admin(request, db)
    if staff:
        return RedirectResponse(url="/staff/queue", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="staff/login.html",
        context={"current_user": user, "current_staff": None, "current_admin": admin, "error": None}
    )

@router.post("/login")
async def handle_staff_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    email_clean = email.strip().lower()
    stmt = select(Staff).where(Staff.email == email_clean)
    res = await db.execute(stmt)
    staff = res.scalar_one_or_none()
    
    if not staff or not verify_password(password, staff.password_hash):
        user = await get_current_user(request, db)
        admin = await get_current_admin(request, db)
        return templates.TemplateResponse(
            request=request,
            name="staff/login.html",
            context={"current_user": user, "current_staff": None, "current_admin": admin, "error": "Invalid staff email or password."}
        )
        
    request.session["staff_id"] = staff.id
    return RedirectResponse(url="/staff/queue", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/logout")
async def handle_staff_logout(request: Request):
    request.session.pop("staff_id", None)
    return RedirectResponse(url="/staff/login", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/queue", response_class=HTMLResponse)
@router.get("/queue/{category_str}", response_class=HTMLResponse)
async def staff_queue_page(
    request: Request,
    category_str: Optional[str] = None,
    category: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    staff = await get_current_staff(request, db)
    user = await get_current_user(request, db)
    admin = await get_current_admin(request, db)
    
    if not staff:
        return RedirectResponse(url="/staff/login", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
        
    cat_filter = None
    target_cat_str = category or category_str
    if target_cat_str and target_cat_str.lower() != "all":
        try:
            cat_filter = CategoryEnum(target_cat_str.capitalize())
        except ValueError:
            pass
            
    complaints = await get_all_active_queue(db, category_filter=cat_filter)
    return templates.TemplateResponse(
        request=request,
        name="staff/queue.html",
        context={
            "current_user": user,
            "current_staff": staff,
            "current_admin": admin,
            "current_category": cat_filter.value if cat_filter else None,
            "category": cat_filter.value if cat_filter else "Master Queue",
            "complaints": complaints
        }
    )

@router.post("/assign/{complaint_id}")
async def assign_complaint_to_self(
    complaint_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    staff = await require_staff(request, db)
    complaint = await db.get(Complaint, complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
        
    complaint.assigned_staff_id = staff.id
    complaint.assigned_staff_name = staff.name
    if complaint.status == StatusEnum.SUBMITTED:
        complaint.status = StatusEnum.ASSIGNED
        
    await db.commit()
    await db.refresh(complaint)
    
    return RedirectResponse(url="/staff/queue", status_code=status.HTTP_303_SEE_OTHER)

@router.post("/status/{complaint_id}")
async def update_complaint_status(
    complaint_id: int,
    request: Request,
    status_str: str = Form(None, alias="status"),
    db: AsyncSession = Depends(get_db)
):
    staff = await require_staff(request, db)
    complaint = await db.get(Complaint, complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
        
    try:
        new_status = StatusEnum(status_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status_str}")
        
    complaint.status = new_status
    if not complaint.assigned_staff_id:
        complaint.assigned_staff_id = staff.id
        complaint.assigned_staff_name = staff.name
        
    await db.commit()
    await db.refresh(complaint)
    
    return RedirectResponse(url="/staff/queue", status_code=status.HTTP_303_SEE_OTHER)
