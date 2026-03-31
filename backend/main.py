"""
backend/main.py
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.api.routes import upload, report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khoi dong: warm up ChromaDB connection truoc request dau tien
    # de tranh cold start lat khi user upload file
    from backend.core.rag import _get_chroma_client
    _get_chroma_client()
    logger.info("ChromaDB warmup finished")
    yield
    # Shutdown: khong can cleanup gi them


app = FastAPI(
    title="GreenCalyx AI",
    description="ESG scoring for logistics documents",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Gradio frontend same-host, doi sang origin cu the khi production
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

app.include_router(upload.router, prefix="/api/v1")
app.include_router(report.router, prefix="/api/v1")

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": settings.app_env}

# Mount thu muc frontend de serve qua HTTP (ho tro truy cap tu LAN)
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="frontend")

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/app.html")