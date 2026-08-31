import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import BASE_DIR, UPLOAD_DIR
from app.database import init_db
from app.main import seed_demo_accounts

async def reset_database():
    db_path = BASE_DIR / "infrapulse.db"
    if db_path.exists():
        os.remove(db_path)
        print(f"[+] Removed existing database: {db_path}")

    # Clear uploaded images except .gitkeep
    for item in UPLOAD_DIR.glob("*"):
        if item.is_file() and item.name != ".gitkeep":
            os.remove(item)
            print(f"[+] Removed upload: {item.name}")

    await init_db()
    await seed_demo_accounts()
    print("[+] Database reset & demo accounts seeded successfully!")
    print("    • Demo User:  user@infrapulse.org       | Password: user123")
    print("    • Admin:      admin@infrapulse.org      | Password: admin123")
    print("    • Staff Structural: structural@infrapulse.org | Password: staff123")
    print("    • Staff Functional: functional@infrapulse.org | Password: staff123")
    print("    • Staff Performance: performance@infrapulse.org | Password: staff123")

if __name__ == "__main__":
    asyncio.run(reset_database())
