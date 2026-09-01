import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, Request, Form, UploadFile, File, HTTPException, Query, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from app.config import BASE_DIR, UPLOAD_DIR, ALLOWED_EXTENSIONS
from app.database import get_db
from app.models import User, Complaint, TicketComment, CategoryEnum, StatusEnum, generate_10_digit_id
from app.auth import hash_password, verify_password, get_current_user, get_current_staff, get_current_admin, require_user
from app.priority_queue import compute_priority_score, get_queue_position, mock_classify_defect

from app.templates_config import templates

router = APIRouter(tags=["User Portal"])

@router.get("/user/register", response_class=HTMLResponse)
async def register_page(request: Request, next: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    if user:
        return RedirectResponse(url=next or "/user/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="user/register.html",
        context={"current_user": None, "current_staff": staff, "current_admin": admin, "next": next, "error": None}
    )

@router.post("/user/register")
async def handle_user_register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(None),
    password: str = Form(...),
    next: Optional[str] = Form(None),
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
            context={"error": "An account with this email address already exists. Please login instead.", "next": next}
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
    target_url = next if (next and next.startswith("/")) else "/user/dashboard"
    return RedirectResponse(url=target_url, status_code=status.HTTP_303_SEE_OTHER)

@router.get("/user/login", response_class=HTMLResponse)
async def login_page(request: Request, next: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    if user:
        return RedirectResponse(url=next or "/user/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="user/login.html",
        context={"current_user": None, "current_staff": staff, "current_admin": admin, "next": next, "error": None}
    )

@router.post("/user/login")
async def handle_user_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: Optional[str] = Form(None),
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
            context={"error": "Invalid user email or password. Please try again.", "next": next}
        )
        
    request.session["user_id"] = user.id
    target_url = next if (next and next.startswith("/")) else "/user/dashboard"
    return RedirectResponse(url=target_url, status_code=status.HTTP_303_SEE_OTHER)

@router.get("/user/logout")
async def handle_user_logout(request: Request):
    request.session.pop("user_id", None)
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/user/submit", response_class=HTMLResponse)
async def submit_form_page(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/user/login?next=/user/submit", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
        
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    return templates.TemplateResponse(
        request=request,
        name="user/submit.html",
        context={"current_user": user, "current_staff": staff, "current_admin": admin, "error": None}
    )

@router.post("/user/submit")
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
    if not user:
        return RedirectResponse(url="/user/login?next=/user/submit", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    try:
        from PIL import Image
        import io

        content = await photo.read()
        image = Image.open(io.BytesIO(content))
        
        # Convert any format (JPEG, WEBP, BMP, GIF, etc.) to RGB and save as PNG
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
            
        unique_filename = f"{uuid.uuid4().hex}.png"
        file_path = UPLOAD_DIR / unique_filename
        image.save(file_path, format="PNG")
        
        relative_photo_url = f"/static/uploads/{unique_filename}"
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="user/submit.html",
            context={"current_user": user, "error": f"Invalid image file. Could not process image: {str(e)}"}
        )

    # Run deep learning vision inference on uploaded defect photo
    from app.model_service import predict_single_image
    pred = predict_single_image(str(file_path))
    
    defect_name = pred["defect_name"]
    category = pred["category"]
    severity = pred["severity"]
    extent = pred["extent"]
    
    priority_score = compute_priority_score(category, defect_name, severity, extent)
    ticket_id = generate_10_digit_id()
    
    complaint = Complaint(
        id=ticket_id,
        user_id=user.id,
        user_name=user_name.strip() or user.name,
        user_email=user_email.strip().lower() or user.email,
        user_phone=user_phone.strip() if user_phone else user.phone,
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
    
    return RedirectResponse(url=f"/ticket/{complaint.id}", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/ticket/{ticket_id}", response_class=HTMLResponse)
@router.get("/user/ticket/{ticket_id}", response_class=HTMLResponse)
async def view_dedicated_ticket_page(ticket_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    complaint = await db.get(Complaint, ticket_id)
    if not complaint:
        raise HTTPException(status_code=404, detail=f"Ticket #{ticket_id} not found")
        
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    queue_pos = await get_queue_position(db, complaint)
    
    # Authorized viewer check: Only ticket owner, staff member, or admin can view personal contact info!
    is_authorized_viewer = bool(
        (user and (user.id == complaint.user_id or user.email == complaint.user_email)) or
        staff or
        admin
    )
    
    # Fetch ticket comments
    comments_stmt = select(TicketComment).where(TicketComment.ticket_id == ticket_id).order_by(TicketComment.created_at.asc())
    res_comments = await db.execute(comments_stmt)
    comments = list(res_comments.scalars().all())

    return templates.TemplateResponse(
        request=request,
        name="user/ticket_detail.html",
        context={
            "current_user": user,
            "current_staff": staff,
            "current_admin": admin,
            "complaint": complaint,
            "queue_position": queue_pos,
            "is_authorized_viewer": is_authorized_viewer,
            "comments": comments
        }
    )

@router.post("/ticket/{ticket_id}/comment")
async def post_ticket_comment(
    ticket_id: int,
    request: Request,
    message: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    complaint = await db.get(Complaint, ticket_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Ticket not found")
        
    user = await get_current_user(request, db)
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    
    if staff:
        sender_name = staff.name
        sender_role = "Staff"
    elif admin:
        sender_name = admin.name
        sender_role = "Admin"
    elif user:
        sender_name = user.name
        sender_role = "User"
    else:
        return RedirectResponse(url=f"/user/login?next=/ticket/{ticket_id}", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
        
    comment = TicketComment(
        ticket_id=ticket_id,
        sender_name=sender_name,
        sender_role=sender_role,
        message=message.strip()
    )
    db.add(comment)
    await db.commit()
    
    return RedirectResponse(url=f"/ticket/{ticket_id}", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/user/dashboard", response_class=HTMLResponse)
async def user_dashboard(
    request: Request,
    status_filter: Optional[str] = Query(None, alias="status"),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    user = await require_user(request, db)
    staff = await get_current_staff(request, db)
    admin = await get_current_admin(request, db)
    
    stmt = select(Complaint).where(
        or_(Complaint.user_id == user.id, Complaint.user_email == user.email)
    )
    
    if status_filter and status_filter.lower() != "all":
        try:
            st_enum = StatusEnum(status_filter)
            stmt = stmt.where(Complaint.status == st_enum)
        except ValueError:
            pass
            
    if search and search.strip():
        q = f"%{search.strip().lower()}%"
        if search.strip().isdigit():
            stmt = stmt.where(or_(Complaint.id == int(search.strip()), Complaint.address.ilike(q)))
        else:
            stmt = stmt.where(or_(Complaint.address.ilike(q), Complaint.description.ilike(q), Complaint.defect_name.ilike(q)))

    stmt = stmt.order_by(Complaint.created_at.desc())
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
            "complaints": complaints,
            "current_status": status_filter,
            "search_query": search or ""
        }
    )
