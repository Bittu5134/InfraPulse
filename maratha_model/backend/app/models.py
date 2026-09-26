import enum
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


def gen_id():
    return str(uuid.uuid4())


class Role(str, enum.Enum):
    user = "user"
    staff = "staff"


class Category(str, enum.Enum):
    structural = "Structural"
    functional = "Functional"
    performance = "Performance"


class Status(str, enum.Enum):
    submitted = "Submitted"
    assigned = "Assigned"
    in_progress = "In Progress"
    resolved = "Resolved"


# Ordinal priority for defect *type* — higher = more urgent.
# Cracked tiles must outrank paint peeling within Performance (per PS).
DEFECT_TYPE_PRIORITY = {
    "spalling": 100,
    "stagnant_water": 100,
    "cracked_tiles": 80,
    "paint_peeling": 40,
}

DEFECT_TO_CATEGORY = {
    "spalling": Category.structural,
    "stagnant_water": Category.functional,
    "cracked_tiles": Category.performance,
    "paint_peeling": Category.performance,
}


class Account(Base):
    __tablename__ = "accounts"

    id = Column(String, primary_key=True, default=gen_id)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(Role), nullable=False)
    # only set for staff accounts
    staff_category = Column(Enum(Category), nullable=True)

    complaints = relationship("Complaint", back_populates="owner")


class Complaint(Base):
    __tablename__ = "complaints"

    id = Column(String, primary_key=True, default=gen_id)
    owner_id = Column(String, ForeignKey("accounts.id"), nullable=False)

    name = Column(String, nullable=False)
    address = Column(String, nullable=False)
    description = Column(String, nullable=True)
    image_path = Column(String, nullable=False)

    defect_label = Column(String, nullable=False)  # e.g. "spalling"
    category = Column(Enum(Category), nullable=False)
    confidence = Column(Float, nullable=False)
    severity_score = Column(Float, nullable=False)  # 0-100, visible extent/severity
    type_priority = Column(Float, nullable=False)   # ordinal defect-type priority

    status = Column(Enum(Status), nullable=False, default=Status.submitted)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("Account", back_populates="complaints")

    @property
    def rank_key(self):
        # sort DESC by (type_priority, severity_score) -> most urgent first
        return (self.type_priority, self.severity_score)
