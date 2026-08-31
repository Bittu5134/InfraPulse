import enum
import random
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, BigInteger, String, Float, Enum, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

def generate_10_digit_id():
    return random.randint(1000000000, 9999999999)

class CategoryEnum(str, enum.Enum):
    STRUCTURAL = "Structural"
    FUNCTIONAL = "Functional"
    PERFORMANCE = "Performance"

class StatusEnum(str, enum.Enum):
    SUBMITTED = "Submitted"
    ASSIGNED = "Assigned"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    phone = Column(String(20), nullable=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    complaints = relationship("Complaint", back_populates="user", cascade="all, delete-orphan")

class Staff(Base):
    __tablename__ = "staff_members"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    domain = Column(Enum(CategoryEnum), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    assigned_complaints = relationship("Complaint", back_populates="assigned_staff")

class Admin(Base):
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

class Complaint(Base):
    __tablename__ = "complaints"

    # 10-digit random Ticket ID
    id = Column(BigInteger, primary_key=True, index=True, default=generate_10_digit_id)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    user_name = Column(String(100), nullable=False)
    user_email = Column(String(150), nullable=False)
    user_phone = Column(String(20), nullable=True)
    address = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    photo_path = Column(String(255), nullable=False)
    
    # Model classification & severity fields
    defect_name = Column(String(100), nullable=True, default="Pending Analysis")
    category = Column(Enum(CategoryEnum), nullable=True)
    severity = Column(Float, default=1.0)
    extent = Column(Float, default=1.0)
    priority_score = Column(Float, default=0.0)
    
    # Lifecycle status & Staff Assignment
    status = Column(Enum(StatusEnum), default=StatusEnum.SUBMITTED, nullable=False)
    assigned_staff_id = Column(Integer, ForeignKey("staff_members.id"), nullable=True)
    assigned_staff_name = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    user = relationship("User", back_populates="complaints")
    assigned_staff = relationship("Staff", back_populates="assigned_complaints")
