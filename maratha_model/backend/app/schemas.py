from datetime import datetime
from pydantic import BaseModel, EmailStr

from .models import Role, Category, Status


class SignupUser(BaseModel):
    name: str
    email: EmailStr
    password: str


class SignupStaff(BaseModel):
    name: str
    email: EmailStr
    password: str
    category: Category


class Login(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    category: Category | None = None


class ComplaintOut(BaseModel):
    id: str
    name: str
    address: str
    description: str | None
    image_path: str
    defect_label: str
    category: Category
    confidence: float
    severity_score: float
    type_priority: float
    status: Status
    created_at: datetime
    queue_position: int | None = None

    class Config:
        from_attributes = True


class StatusUpdate(BaseModel):
    status: Status
