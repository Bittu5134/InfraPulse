"""
Resets the database to a clean demo state: wipes all accounts and
complaints, then creates fixed demo credentials (3 users, 4 staff).

Usage:
    python -m app.seed
"""
from .auth import hash_password
from .database import Base, SessionLocal, engine
from .models import Account, Category, Complaint, Role

DEMO_PASSWORD = "demo1234"

DEMO_USERS = [
    {"name": "Aditi Sharma", "email": "user1@infrapulse.demo"},
    {"name": "Rohan Mehta", "email": "user2@infrapulse.demo"},
    {"name": "Priya Nair", "email": "user3@infrapulse.demo"},
]

DEMO_STAFF = [
    {"name": "Structural Staff", "email": "staff.structural@infrapulse.demo", "category": Category.structural},
    {"name": "Functional Staff", "email": "staff.functional@infrapulse.demo", "category": Category.functional},
    {"name": "Performance Staff A", "email": "staff.performance1@infrapulse.demo", "category": Category.performance},
    {"name": "Performance Staff B", "email": "staff.performance2@infrapulse.demo", "category": Category.performance},
]


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(Complaint).delete()
        db.query(Account).delete()
        db.commit()

        for u in DEMO_USERS:
            db.add(Account(
                name=u["name"],
                email=u["email"],
                hashed_password=hash_password(DEMO_PASSWORD),
                role=Role.user,
            ))

        for s in DEMO_STAFF:
            db.add(Account(
                name=s["name"],
                email=s["email"],
                hashed_password=hash_password(DEMO_PASSWORD),
                role=Role.staff,
                staff_category=s["category"],
            ))

        db.commit()
    finally:
        db.close()

    print(f"Seeded {len(DEMO_USERS)} users and {len(DEMO_STAFF)} staff accounts.")
    print(f"All passwords: {DEMO_PASSWORD}\n")
    print("Users:")
    for u in DEMO_USERS:
        print(f"  {u['email']}")
    print("Staff:")
    for s in DEMO_STAFF:
        print(f"  {s['email']}  ({s['category'].value})")


if __name__ == "__main__":
    main()
