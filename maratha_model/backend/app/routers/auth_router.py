from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import schemas
from ..auth import create_access_token, hash_password, verify_password
from ..database import get_db
from ..models import Account, Role

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup/user", response_model=schemas.Token)
def signup_user(data: schemas.SignupUser, db: Session = Depends(get_db)):
    if db.query(Account).filter(Account.email == data.email).first():
        raise HTTPException(400, "Email already registered")
    account = Account(
        name=data.name,
        email=data.email,
        hashed_password=hash_password(data.password),
        role=Role.user,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    token = create_access_token({"sub": account.id, "role": account.role.value})
    return schemas.Token(access_token=token, role=account.role)


@router.post("/signup/staff", response_model=schemas.Token)
def signup_staff(data: schemas.SignupStaff, db: Session = Depends(get_db)):
    if db.query(Account).filter(Account.email == data.email).first():
        raise HTTPException(400, "Email already registered")
    account = Account(
        name=data.name,
        email=data.email,
        hashed_password=hash_password(data.password),
        role=Role.staff,
        staff_category=data.category,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    token = create_access_token({"sub": account.id, "role": account.role.value})
    return schemas.Token(access_token=token, role=account.role, category=account.staff_category)


@router.post("/login", response_model=schemas.Token)
def login(data: schemas.Login, db: Session = Depends(get_db)):
    account = db.query(Account).filter(Account.email == data.email).first()
    if not account or not verify_password(data.password, account.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": account.id, "role": account.role.value})
    return schemas.Token(access_token=token, role=account.role, category=account.staff_category)
