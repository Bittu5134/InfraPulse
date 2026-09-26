import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import Base, engine
from .routers import auth_router, complaints_router, staff_router
from .ws import manager

Base.metadata.create_all(bind=engine)

app = FastAPI(title="InfraPulse API")

# ALLOWED_ORIGINS env var: comma-separated list of allowed origins.
# Set this in your Hugging Face Space secrets to your frontend URL.
# Defaults to "*" for local development.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

app.include_router(auth_router.router)
app.include_router(complaints_router.router)
app.include_router(staff_router.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/queue")
async def queue_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # client doesn't need to send anything; keep-alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
