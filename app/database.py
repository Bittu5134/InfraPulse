from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
from app.config import DATABASE_URL

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # SQLite schema auto-migration for complaints table columns if DB already existed
        if "sqlite" in DATABASE_URL:
            res = await conn.execute(text("PRAGMA table_info(complaints)"))
            columns = [row[1] for row in res.fetchall()]
            
            missing_cols = {
                "user_id": "INTEGER REFERENCES users(id)",
                "user_email": "VARCHAR(150)",
                "user_phone": "VARCHAR(20)",
                "assigned_staff_id": "INTEGER REFERENCES staff_members(id)",
                "assigned_staff_name": "VARCHAR(100)"
            }
            
            for col_name, col_type in missing_cols.items():
                if col_name not in columns:
                    await conn.execute(text(f"ALTER TABLE complaints ADD COLUMN {col_name} {col_type}"))
