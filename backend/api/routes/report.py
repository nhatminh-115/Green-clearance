"""
api/routes/report.py

GET /api/v1/reports          — lay danh sach report gan nhat tu Supabase
GET /api/v1/reports/{id}     — lay chi tiet mot report
"""

import logging
from fastapi import APIRouter, HTTPException
from supabase import create_client

from backend.config import get_settings

log = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

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
            .select(
                "id, filename, lane, score, created_at, "
                "total_co2e_kg, transport_co2e_kg, packaging_co2e_kg, "
                "origin_port, destination_port, transport_mode, "
                "distance_km, cargo_weight_tons, upload_mode, "
                "conflict_count, files_uploaded"
            )
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows: list[dict] = list(response.data or []) #type: ignore[assignment]
        return rows
    except Exception as e:
        log.exception(f"Supabase list loi: {e}")
        raise HTTPException(status_code=500, detail="Cannot load report list.")


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
            raise HTTPException(status_code=404, detail="Report not found.")
        row: dict = response.data or {} #type: ignore[assignment]
        if not row:
            raise HTTPException(status_code=404, detail="Report not found.")
        return row
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Supabase get loi: {e}")
        raise HTTPException(status_code=500, detail="Cannot load report.")