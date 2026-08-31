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

    id = Column(BigInteger, primary_key=True, index=True, default=generate_10_digit_id)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    user_name = Column(String(100), nullable=False)
    user_email = Column(String(150), nullable=False)
    user_phone = Column(String(20), nullable=True)
    address = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    photo_path = Column(String(255), nullable=False)
    
    defect_name = Column(String(100), nullable=True, default="Pending Analysis")
    category = Column(Enum(CategoryEnum), nullable=True)
    severity = Column(Float, default=1.0)
    extent = Column(Float, default=1.0)
    priority_score = Column(Float, default=0.0)
    
    status = Column(Enum(StatusEnum), default=StatusEnum.SUBMITTED, nullable=False)
    assigned_staff_id = Column(Integer, ForeignKey("staff_members.id"), nullable=True)
    assigned_staff_name = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    user = relationship("User", back_populates="complaints")
    assigned_staff = relationship("Staff", back_populates="assigned_complaints")
    comments = relationship("TicketComment", back_populates="ticket", cascade="all, delete-orphan", order_by="TicketComment.created_at.asc()")

class TicketComment(Base):
    __tablename__ = "ticket_comments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(BigInteger, ForeignKey("complaints.id"), nullable=False)
    sender_name = Column(String(100), nullable=False)
    sender_role = Column(String(50), nullable=False)  # "User", "Staff", "Admin"
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    ticket = relationship("Complaint", back_populates="comments")
