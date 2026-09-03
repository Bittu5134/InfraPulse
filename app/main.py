import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import select
from app.config import UPLOAD_DIR, BASE_DIR, SECRET_KEY
from app.database import init_db, AsyncSessionLocal
from app.models import User, Staff, Admin, CategoryEnum
from app.auth import hash_password, get_current_user, get_current_staff, get_current_admin
from app.templates_config import templates
from app.routers import user, staff, admin, api, live, test_bench

async def seed_demo_accounts():
    async with AsyncSessionLocal() as session:
        # Seed Admin Account
        admin_stmt = select(Admin).where(Admin.email == "admin@infrapulse.org")
        if not (await session.execute(admin_stmt)).scalar_one_or_none():
            admin_user = Admin(
                name="System Administrator",
                email="admin@infrapulse.org",
                password_hash=hash_password("admin123")
            )
            session.add(admin_user)

        # Seed Staff Accounts
        demo_staff = [
            ("Structural Staff", "structural@infrapulse.org", "staff123", CategoryEnum.STRUCTURAL),
            ("Functional Staff", "functional@infrapulse.org", "staff123", CategoryEnum.FUNCTIONAL),
            ("Performance Staff", "performance@infrapulse.org", "staff123", CategoryEnum.PERFORMANCE),
        ]
        for name, email, password, domain in demo_staff:
            stmt = select(Staff).where(Staff.email == email)
            res = await session.execute(stmt)
            if not res.scalar_one_or_none():
                staff_member = Staff(
                    name=name,
                    email=email,
                    password_hash=hash_password(password),
                    domain=domain
                )
                session.add(staff_member)

        # Seed Demo Testing User Account
        user_stmt = select(User).where(User.email == "user@infrapulse.org")
        if not (await session.execute(user_stmt)).scalar_one_or_none():
            demo_user = User(
                name="Demo User",
                email="user@infrapulse.org",
                phone="9876543210",
                password_hash=hash_password("user123")
            )
            session.add(demo_user)

        await session.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    await seed_demo_accounts()
    yield

app = FastAPI(
    title="InfraPulse - Defect Priority Maintenance System",
    description="Automated defect priority detection and ticketing tool.",
    version="3.0.0",
    lifespan=lifespan
)

# Custom 404 Error Exception Handler
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        async with AsyncSessionLocal() as db:
            u = await get_current_user(request, db)
            s = await get_current_staff(request, db)
            a = await get_current_admin(request, db)
        return templates.TemplateResponse(
            request=request,
            name="404.html",
            context={"current_user": u, "current_staff": s, "current_admin": a},
            status_code=404
        )
    return HTMLResponse(content=str(exc.detail), status_code=exc.status_code)

# Middleware to disable all HTTP caching across all routes, static assets, and APIs
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Proxy Headers Middleware for reverse proxies (Cloudflare / Railway)
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Session Middleware for User, Staff, Admin Auth
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Mount static directory
static_dir = BASE_DIR / "app" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount dataset images for testing & benchmarking
model_data_dir = BASE_DIR / "app" / "model" / "data"
if model_data_dir.exists():
    app.mount("/test_data", StaticFiles(directory=str(model_data_dir)), name="test_data")

# Include Routers
app.include_router(user.router)
app.include_router(staff.router)
app.include_router(admin.router)
app.include_router(api.router)
app.include_router(live.router)
app.include_router(test_bench.router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "system": "InfraPulse", "version": "3.0.0"}

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    async with AsyncSessionLocal() as db:
        u = await get_current_user(request, db)
        s = await get_current_staff(request, db)
        a = await get_current_admin(request, db)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"current_user": u, "current_staff": s, "current_admin": a}
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
