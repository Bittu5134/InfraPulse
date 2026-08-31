import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "app" / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR}/infrapulse.db")

SECRET_KEY = os.getenv("SECRET_KEY", "infrapulse-secret-key-change-in-production")

# Defect Category Priority Weights (Structural is highest, then Functional, then Performance)
CATEGORY_WEIGHTS = {
    "Structural": 3.0,
    "Functional": 2.0,
    "Performance": 1.0,
}

# Defect specific priority score boost
# cracked tiles > paint peeling per PDF requirement
DEFECT_PRIORITY_BOOST = {
    "spalling": 10.0,
    "stagnant water": 6.0,
    "cracked tiles": 4.0,
    "paint peeling": 2.0,
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
