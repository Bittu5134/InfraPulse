import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, Request, Form, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import BASE_DIR, UPLOAD_DIR, ALLOWED_EXTENSIONS
from app.database import get_db
from app.models import User, Complaint, CategoryEnum, StatusEnum
from app.auth import hash_password, verify_password, get_current_user, get_current_staff, get_current_admin
from app.priority_queue import compute_priority_score, get_queue_position, mock_classify_defect

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

router = APIRouter(prefix="/user", tags=["User Portal"])

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    return templates.TemplateResponse(
        request=request,
        name="user/register.html",
        context={"current_user": user, "current_staff": staff, "current_admin": admin, "error": None}
    )

@router.post("/register")
async def handle_user_register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(None),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    email_clean = email.strip().lower()
    stmt = select(User).where(User.email == email_clean)
    res = await db.execute(stmt)
    existing = res.scalar_one_or_none()
    
    if existing:
        return templates.TemplateResponse(
            request=request,
            name="user/register.html",
            context={"error": "An account with this email already exists."}
        )
        
    user = User(
        name=name.strip(),
        email=email_clean,
        phone=phone.strip() if phone else None,
        password_hash=hash_password(password)
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    request.session["user_id"] = user.id
    return RedirectResponse(url="/user/dashboard", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    return templates.TemplateResponse(
        request=request,
        name="user/login.html",
        context={"current_user": user, "current_staff": staff, "current_admin": admin, "error": None}
    )

@router.post("/login")
async def handle_user_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    email_clean = email.strip().lower()
    stmt = select(User).where(User.email == email_clean)
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()
    
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            request=request,
            name="user/login.html",
            context={"error": "Invalid email or password."}
        )
        
    request.session["user_id"] = user.id
    return RedirectResponse(url="/user/dashboard", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/logout")
async def handle_user_logout(request: Request):
    request.session.pop("user_id", None)
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/submit", response_class=HTMLResponse)
async def submit_form_page(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    return templates.TemplateResponse(
        request=request,
        name="user/submit.html",
        context={"current_user": user, "current_staff": staff, "current_admin": admin, "error": None}
    )

@router.post("/submit")
async def handle_complaint_submit(
    request: Request,
    user_name: str = Form(...),
    user_email: str = Form(...),
    user_phone: str = Form(None),
    address: str = Form(...),
    description: str = Form(...),
    photo: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    user = await get_current_user(request, db)
    ext = Path(photo.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse(
            request=request,
            name="user/submit.html",
            context={"current_user": user, "error": f"Invalid image format '{ext}'. Supported: JPG, PNG, WEBP."}
        )
        
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    content = await photo.read()
    with open(file_path, "wb") as f:
        f.write(content)
        
    relative_photo_url = f"/static/uploads/{unique_filename}"
    
    auto_class = mock_classify_defect(description, photo.filename)
    defect_name = auto_class["defect_name"]
    category = auto_class["category"]
    severity = auto_class["severity"]
    extent = auto_class["extent"]
    
    priority_score = compute_priority_score(category, defect_name, severity, extent)
    
    complaint = Complaint(
        user_id=user.id if user else None,
        user_name=user_name.strip(),
        user_email=user_email.strip().lower(),
        user_phone=user_phone.strip() if user_phone else None,
        address=address.strip(),
        description=description.strip(),
        photo_path=relative_photo_url,
        defect_name=defect_name,
        category=category,
        severity=severity,
        extent=extent,
        priority_score=priority_score,
        status=StatusEnum.SUBMITTED
    )
    
    db.add(complaint)
    await db.commit()
    await db.refresh(complaint)
    
    if user:
        return RedirectResponse(url="/user/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url=f"/user/dashboard?email={user_email.strip().lower()}", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request, email: str = None, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    
    query_email = user.email if user else (email.strip().lower() if email else None)
    
    complaints = []
    if query_email:
        stmt = select(Complaint).where(Complaint.user_email == query_email).order_by(Complaint.created_at.desc())
        res = await db.execute(stmt)
        complaints = list(res.scalars().all())
        
        for c in complaints:
            c.queue_position = await get_queue_position(db, c)
            
    return templates.TemplateResponse(
        request=request,
        name="user/dashboard.html",
        context={
            "current_user": user,
            "current_staff": staff,
            "current_admin": admin,
            "complaints": complaints
        }
    )
