import pytest
import pytest_asyncio
from io import BytesIO
from PIL import Image
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select

from app.main import app
from app.database import Base, get_db
from app.models import User, Staff, Admin, Complaint, CategoryEnum, StatusEnum
from app.auth import hash_password
from app.priority_queue import compute_priority_score

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)

async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session

app.dependency_overrides[get_db] = override_get_db

def create_test_png():
    buf = BytesIO()
    img = Image.new("RGB", (20, 20), color="blue")
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

@pytest_asyncio.fixture(autouse=True)
async def init_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestingSessionLocal() as session:
        staff1 = Staff(
            name="Alice Structural",
            email="alice@infrapulse.org",
            password_hash=hash_password("staff123"),
            domain=CategoryEnum.STRUCTURAL
        )
        admin1 = Admin(
            name="Admin User",
            email="admin@infrapulse.org",
            password_hash=hash_password("admin123")
        )
        session.add(staff1)
        session.add(admin1)
        await session.commit()
        
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.mark.asyncio
async def test_priority_score_computation():
    score_struct = compute_priority_score(CategoryEnum.STRUCTURAL, "Spalling", severity=8.0, extent=50.0)
    score_perf_tiles = compute_priority_score(CategoryEnum.PERFORMANCE, "Cracked Tiles", severity=8.0, extent=50.0)
    score_perf_paint = compute_priority_score(CategoryEnum.PERFORMANCE, "Paint Peeling", severity=8.0, extent=50.0)
    
    assert score_struct > score_perf_tiles
    assert score_perf_tiles > score_perf_paint

@pytest.mark.asyncio
async def test_user_registration_login_and_ticket_submission():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        reg_resp = await client.post(
            "/user/register",
            data={
                "name": "Bob Builder",
                "email": "bob@example.com",
                "phone": "+91 9876543210",
                "password": "user123"
            },
            follow_redirects=True
        )
        assert reg_resp.status_code == 200

        fake_file = create_test_png()
        sub_resp = await client.post(
            "/user/submit",
            data={
                "user_name": "Bob Builder",
                "user_email": "bob@example.com",
                "user_phone": "+91 9876543210",
                "address": "Block 5, Flat 12",
                "description": "Concrete spalling on main beam"
            },
            files={"photo": ("spalling.png", fake_file, "image/png")},
            follow_redirects=True
        )
        assert sub_resp.status_code == 200
        assert "Bob Builder" in sub_resp.text or "Spalling" in sub_resp.text

@pytest.mark.asyncio
async def test_staff_login_and_self_assignment():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        await client.post(
            "/user/register",
            data={"name": "Charlie", "email": "charlie@example.com", "password": "pass"},
            follow_redirects=True
        )
        fake_file = create_test_png()
        sub_resp = await client.post(
            "/user/submit",
            data={
                "user_name": "Charlie",
                "user_email": "charlie@example.com",
                "address": "Hall 3",
                "description": "Concrete spalling structural issue"
            },
            files={"photo": ("photo.png", fake_file, "image/png")},
            follow_redirects=True
        )
        assert sub_resp.status_code == 200
        
        unauth_resp = await client.get("/staff/queue", follow_redirects=False)
        assert unauth_resp.status_code in [303, 307]

        login_resp = await client.post(
            "/staff/login",
            data={"email": "alice@infrapulse.org", "password": "staff123"},
            follow_redirects=True
        )
        assert login_resp.status_code == 200
        assert "Staff Maintenance Portal" in login_resp.text

        async with TestingSessionLocal() as session:
            stmt = select(Complaint).where(Complaint.user_email == "charlie@example.com")
            res = await session.execute(stmt)
            created_ticket = res.scalar_one()

            # Ensure staff domain matches the ticket category
            staff_stmt = select(Staff).where(Staff.email == "alice@infrapulse.org")
            alice = (await session.execute(staff_stmt)).scalar_one()
            alice.domain = created_ticket.category
            await session.commit()

        assign_resp = await client.post(f"/staff/assign/{created_ticket.id}", follow_redirects=True)
        assert assign_resp.status_code == 200
        assert "Alice Structural" in assign_resp.text

@pytest.mark.asyncio
async def test_admin_portal_management():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        login_resp = await client.post(
            "/admin/login",
            data={"email": "admin@infrapulse.org", "password": "admin123"},
            follow_redirects=True
        )
        assert login_resp.status_code == 200
        assert "Admin" in login_resp.text

        create_staff_resp = await client.post(
            "/admin/staff/create",
            data={
                "name": "New Tech Staff",
                "email": "newtech@infrapulse.org",
                "password": "staffpass123",
                "domain": "Functional"
            },
            follow_redirects=True
        )
        assert create_staff_resp.status_code == 200
        assert "newtech@infrapulse.org" in create_staff_resp.text

@pytest.mark.asyncio
async def test_benchmark_page():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/test?page=1&page_size=5")
        assert resp.status_code == 200
        assert "Multi-Model" in resp.text
        assert "Leaderboard" in resp.text
        assert "ConvNeXt" in resp.text or "EfficientNet" in resp.text
