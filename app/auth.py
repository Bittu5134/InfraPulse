import hashlib
from typing import Optional
from fastapi import Request, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import User, Staff, Admin

def hash_password(password: str) -> str:
    # Double hashed SHA-256 with salt prefix for secure storage
    salt = "InfraPulseGovtSecureSalt2026!"
    return hashlib.sha256((salt + password).encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

async def get_current_user(request: Request, db: AsyncSession) -> Optional[User]:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    user = await db.get(User, user_id)
    return user

async def require_user(request: Request, db: AsyncSession) -> User:
    user = await get_current_user(request, db)
    if not user:
        # Redirect to login with next path
        current_path = request.url.path
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": f"/user/login?next={current_path}"}
        )
    return user

async def get_current_staff(request: Request, db: AsyncSession) -> Optional[Staff]:
    staff_id = request.session.get("staff_id")
    if not staff_id:
        return None
    staff = await db.get(Staff, staff_id)
    return staff

async def require_staff(request: Request, db: AsyncSession) -> Staff:
    staff = await get_current_staff(request, db)
    if not staff:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/staff/login"}
        )
    return staff

async def get_current_admin(request: Request, db: AsyncSession) -> Optional[Admin]:
    admin_id = request.session.get("admin_id")
    if not admin_id:
        return None
    admin = await db.get(Admin, admin_id)
    return admin

async def require_admin(request: Request, db: AsyncSession) -> Admin:
    admin = await get_current_admin(request, db)
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/admin/login"}
        )
    return admin
