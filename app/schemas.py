from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from app.models import CategoryEnum, StatusEnum

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: Optional[str] = None
    password: str = Field(..., min_length=4)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class StaffCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=4)
    domain: CategoryEnum

class StaffLogin(BaseModel):
    email: EmailStr
    password: str

class ComplaintCreate(BaseModel):
    user_name: str = Field(..., min_length=2, max_length=100)
    user_email: EmailStr
    user_phone: Optional[str] = None
    address: str = Field(..., min_length=2, max_length=255)
    description: str = Field(..., min_length=5)

class ClassificationPayload(BaseModel):
    complaint_id: int
    defect_name: str
    category: CategoryEnum
    severity: float = Field(..., ge=1.0, le=10.0)
    extent: float = Field(..., ge=0.0, le=100.0)

class ComplaintStatusUpdate(BaseModel):
    status: StatusEnum
    assigned_staff_id: Optional[int] = None
    assigned_staff_name: Optional[str] = None

class ComplaintResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: Optional[int]
    user_name: str
    user_email: str
    user_phone: Optional[str]
    address: str
    description: str
    photo_path: str
    defect_name: Optional[str]
    category: Optional[CategoryEnum]
    severity: float
    extent: float
    priority_score: float
    status: StatusEnum
    assigned_staff_id: Optional[int]
    assigned_staff_name: Optional[str]
    created_at: datetime
    updated_at: datetime
    queue_position: Optional[int] = None
