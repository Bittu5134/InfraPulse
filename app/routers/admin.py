from fastapi import APIRouter, Depends, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.config import BASE_DIR
from app.database import get_db
from app.models import Admin, User, Staff, Complaint, CategoryEnum, StatusEnum
from app.auth import hash_password, verify_password, get_current_admin, get_current_user, get_current_staff, require_admin

from app.templates_config import templates

router = APIRouter(prefix="/admin", tags=["Admin Portal"])

@router.get("/login", response_class=HTMLResponse)
async def admin_login_page(request: Request, db: AsyncSession = Depends(get_db)):
    admin = await get_current_admin(request, db)
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    if admin:
        return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="admin/login.html",
        context={"current_user": user, "current_staff": staff, "current_admin": None, "error": None}
    )

@router.post("/login")
async def handle_admin_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    email_clean = email.strip().lower()
    stmt = select(Admin).where(Admin.email == email_clean)
    res = await db.execute(stmt)
    admin = res.scalar_one_or_none()
    
    if not admin or not verify_password(password, admin.password_hash):
        user = await get_current_user(request, db)
        staff = await get_current_staff(request, db)
        return templates.TemplateResponse(
            request=request,
            name="admin/login.html",
            context={"current_user": user, "current_staff": staff, "current_admin": None, "error": "Invalid admin credentials."}
        )
        
    request.session["admin_id"] = admin.id
    return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/logout")
async def handle_admin_logout(request: Request):
    request.session.pop("admin_id", None)
    return RedirectResponse(url="/admin/login", status_code=status.HTTP_303_SEE_OTHER)

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    admin = await require_admin(request, db)
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    
    # Overview counts
    users_count = (await db.execute(select(func.count(User.id)))).scalar_one()
    staff_count = (await db.execute(select(func.count(Staff.id)))).scalar_one()
    tickets_count = (await db.execute(select(func.count(Complaint.id)))).scalar_one()
    pending_count = (await db.execute(select(func.count(Complaint.id)).where(Complaint.status != StatusEnum.RESOLVED))).scalar_one()
    
    # Data lists
    staff_members = (await db.execute(select(Staff).order_by(Staff.domain, Staff.name))).scalars().all()
    users_list = (await db.execute(select(User).order_by(User.created_at.desc()))).scalars().all()
    tickets_list = (await db.execute(select(Complaint).order_by(Complaint.created_at.desc()))).scalars().all()
    
    return templates.TemplateResponse(
        request=request,
        name="admin/dashboard.html",
        context={
            "current_admin": admin,
            "current_user": user,
            "current_staff": staff,
            "metrics": {
                "users": users_count,
                "staff": staff_count,
                "tickets": tickets_count,
                "pending": pending_count
            },
            "staff_members": staff_members,
            "users_list": users_list,
            "tickets_list": tickets_list,
            "categories": [c.value for c in CategoryEnum]
        }
    )

@router.post("/staff/create")
async def create_staff_member(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    domain: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    await require_admin(request, db)
    email_clean = email.strip().lower()
    
    existing = (await db.execute(select(Staff).where(Staff.email == email_clean))).scalar_one_or_none()
    if existing:
        return RedirectResponse(url="/admin?error=Staff+email+already+exists", status_code=status.HTTP_303_SEE_OTHER)
        
    try:
        domain_enum = CategoryEnum(domain)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid domain")
        
    new_staff = Staff(
        name=name.strip(),
        email=email_clean,
        password_hash=hash_password(password),
        domain=domain_enum
    )
    db.add(new_staff)
    await db.commit()
    return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)

@router.post("/staff/delete/{staff_id}")
async def delete_staff_member(staff_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    await require_admin(request, db)
    staff_member = await db.get(Staff, staff_id)
    if staff_member:
        await db.delete(staff_member)
        await db.commit()
    return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)

@router.post("/ticket/delete/{ticket_id}")
async def delete_ticket(ticket_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    await require_admin(request, db)
    ticket = await db.get(Complaint, ticket_id)
    if ticket:
        await db.delete(ticket)
        await db.commit()
    return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
