"""
api/routes/report.py

GET /api/v1/reports          — lay danh sach report gan nhat tu Supabase
GET /api/v1/reports/{id}     — lay chi tiet mot report
"""

import logging
from datetime import datetime, timedelta, timezone
import calendar
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


# ---------------------------------------------------------------------------
# Helper functions for date ranges
# ---------------------------------------------------------------------------
def _get_week_range(now: datetime) -> tuple[datetime, datetime]:
    start = now - timedelta(days=now.weekday())  # Monday
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=7)
    return start, end


def _get_month_range(now: datetime) -> tuple[datetime, datetime]:
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    _, last_day = calendar.monthrange(now.year, now.month)
    end = now.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999)
    return start, end


def _get_year_range(now: datetime) -> tuple[datetime, datetime]:
    start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    end = now.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
    return start, end


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
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
        rows: list[dict] = list(response.data or [])  # type: ignore[assignment]
        return rows
    except Exception as e:
        log.exception(f"Supabase list loi: {e}")
        raise HTTPException(status_code=500, detail="Cannot load report list.")


@router.get("/reports/stats")
def get_report_stats() -> dict:
    try:
        table = _get_supabase().table(settings.supabase_table_reports)

        # Total distinct shipments
        total_response = table.select("shipment_id", count="exact", head=True).execute()
        total_shipments = int(total_response.count or 0)

        # Green, Yellow, Red by distinct shipment_id
        # Supabase doesn't support COUNT(DISTINCT) directly, so we fetch distinct shipment_id and filter.
        # But we can use .select("shipment_id").execute() and count in Python.
        # For simplicity, we'll fetch all shipment_ids and count manually (up to 10k records fine).
        all_shipments = table.select("shipment_id, lane").execute()
        shipments = all_shipments.data or []

        # Count distinct shipment_id per lane
        seen = set()
        green = yellow = red = 0
        for row in shipments:
            sid = row.get("shipment_id")
            lane = row.get("lane")
            if sid is None:
                continue
            if (sid, lane) in seen:
                continue
            seen.add((sid, lane))
            if lane == "GREEN":
                green += 1
            elif lane == "YELLOW":
                yellow += 1
            elif lane == "RED":
                red += 1

        # Get all scores (non-null) for average calculation (still per row, as average of all upload attempts)
        scores_response = (
            table.select("score")
            .not_.is_("score", "null")
            .execute()
        )
        scores = [float(row["score"]) for row in scores_response.data] if scores_response.data else []
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "total_shipments": total_shipments,
            "green": green,
            "yellow": yellow,
            "red": red,
            "average_score": round(avg_score, 2),
        }
    except Exception as e:
        log.exception(f"Supabase stats error: {e}")
        raise HTTPException(status_code=500, detail="Cannot load report statistics.")


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
        row: dict = response.data or {}  # type: ignore[assignment]
        if not row:
            raise HTTPException(status_code=404, detail="Report not found.")
        return row
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Supabase get loi: {e}")
        raise HTTPException(status_code=500, detail="Cannot load report.")


@router.get("/reports/emissions/total")
def get_emissions_totals() -> dict:
    """Get total CO₂e emissions for today, this week, this month, this year."""
    try:
        now = datetime.now(timezone.utc)

        # Today range
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

        week_start, week_end = _get_week_range(now)
        month_start, month_end = _get_month_range(now)
        year_start, year_end = _get_year_range(now)

        supabase = _get_supabase()
        table = supabase.table(settings.supabase_table_reports)

        def fetch_sum(start: datetime, end: datetime) -> float:
            resp = (
                table.select("total_co2e_kg")
                .gte("created_at", start.isoformat())
                .lte("created_at", end.isoformat())
                .execute()
            )
            total = sum(float(row["total_co2e_kg"]) for row in resp.data if row.get("total_co2e_kg") is not None)
            return round(total, 2)

        today_total = fetch_sum(today_start, today_end)
        week_total = fetch_sum(week_start, week_end)
        month_total = fetch_sum(month_start, month_end)
        year_total = fetch_sum(year_start, year_end)

        return {
            "today": today_total if today_total > 0 else 0,
            "this_week": week_total if week_total > 0 else 0,
            "this_month": month_total if month_total > 0 else 0,
            "this_year": year_total if year_total > 0 else 0,
        }
    except Exception as e:
        log.exception(f"Emissions totals error: {e}")
        raise HTTPException(status_code=500, detail="Cannot load emissions totals.")
    
@router.get("/reports/shipments")
def list_shipments(limit: int = 100) -> list[dict]:
    """
    List distinct shipments (grouped by shipment_id) with aggregated data.
    """
    try:
        # We need to get all reports, then group in Python because Supabase doesn't support GROUP BY with aggregates easily.
        # But for simplicity, fetch all reports (up to a limit) and group.
        response = (
            _get_supabase()
            .table(settings.supabase_table_reports)
            .select(
                "id, shipment_id, created_at, total_co2e_kg, score, lane, files_uploaded"
            )
            .order("created_at", desc=True)
            .limit(1000)
            .execute()
        )
        rows = response.data or []

        # Group by shipment_id
        shipments = {}
        for row in rows:
            sid = row.get("shipment_id")
            if not sid:
                continue
            if sid not in shipments:
                shipments[sid] = {
                    "shipment_id": sid,
                    "created_at": row["created_at"],
                    "total_co2e_kg": row.get("total_co2e_kg"),
                    "score": row.get("score"),
                    "lane": row.get("lane"),
                    "files_uploaded": row.get("files_uploaded"),
                    "report_id": row.get("id"),  # reference to one report
                }
            else:
                # Keep earliest or latest? We'll keep the first seen (latest due to order)
                pass

        return list(shipments.values())
    except Exception as e:
        log.exception(f"Supabase shipments list error: {e}")
        raise HTTPException(status_code=500, detail="Cannot load shipments list.")


@router.delete("/reports/shipment/{shipment_id}")
def delete_shipment(shipment_id: str) -> dict:
    """Delete all reports with a given shipment_id."""
    try:
        result = (
            _get_supabase()
            .table(settings.supabase_table_reports)
            .delete()
            .eq("shipment_id", shipment_id)
            .execute()
        )
        return {"status": "deleted", "count": len(result.data or [])}
    except Exception as e:
        log.exception(f"Delete shipment error: {e}")
        raise HTTPException(status_code=500, detail="Cannot delete shipment.")


@router.delete("/reports/shipments/all")
def delete_all_shipments() -> dict:
    """Delete all reports (admin only – maybe restrict later)."""
    try:
        result = (
            _get_supabase()
            .table(settings.supabase_table_reports)
            .delete()
            .neq("id", "none")  # delete all rows
            .execute()
        )
        return {"status": "deleted", "count": len(result.data or [])}
    except Exception as e:
        log.exception(f"Delete all shipments error: {e}")
        raise HTTPException(status_code=500, detail="Cannot delete all shipments.")