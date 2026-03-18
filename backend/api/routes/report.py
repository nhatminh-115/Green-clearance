"""
api/routes/report.py

GET /api/v1/reports          — lay danh sach report gan nhat tu Supabase
GET /api/v1/reports/{id}     — lay chi tiet mot report
"""

import logging
from fastapi import APIRouter, HTTPException
from gradio.monitoring_dashboard import data
from supabase import create_client

from backend.config import get_settings

log = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Supabase client khoi tao lazy — chi tao khi co request dau tien
# tranh loi khi Supabase chua san sang luc app start
_supabase = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(settings.supabase_url, settings.supabase_key)
    return _supabase


@router.get("/reports")
def list_reports(limit: int = 10) -> list[dict]:
    """Lay danh sach report moi nhat, giam dan theo thoi gian."""
    try:
        response = (
            _get_supabase()
            .table(settings.supabase_table_reports)
            .select("id, filename, lane, score, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        data: list[dict] = response.data or []  # type: ignore[assignment]
        return data
    except Exception as e:
        log.exception(f"Supabase list loi: {e}")
        raise HTTPException(status_code=500, detail="Khong the lay danh sach report.")


@router.get("/reports/{report_id}")
def get_report(report_id: str) -> dict:
    """Lay chi tiet mot report theo ID."""
    try:
        response = (
            _get_supabase()
            .table(settings.supabase_table_reports)
            .select("*")
            .eq("id", report_id)
            .single()
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail="Report khong ton tai.")
        data: dict = response.data  # type: ignore[assignment]
        if not data:
            raise HTTPException(status_code=404, detail="Report khong ton tai.")
        return data
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Supabase get loi: {e}")
        raise HTTPException(status_code=500, detail="Khong the lay report.")