"""
api/routes/upload.py

POST /api/v1/upload          — single document (backward compatible)
POST /api/v1/upload/multi    — multi-document batch (CI + PL + BL + optional TDS/PPWR)
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.core.agent import run_pipeline
from backend.core.extractor import extract_and_classify
from backend.core.merger import identify_missing_document_types, merge_documents
from backend.models.schemas import (
    DocumentType, FileAnalysis, MultiDocumentReportResponse,
    ReportResponse,
)
from backend.config import get_settings

log = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB per file
_MAX_FILES_PER_BATCH = 6           # CI + PL + BL + TDS + PPWR + 1 extra

_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}

# Supabase client — lazy init
_supabase = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(settings.supabase_url, settings.supabase_key)
    return _supabase


# ---------------------------------------------------------------------------
# Supabase logging helpers
# ---------------------------------------------------------------------------

def _log_single_to_supabase(
    report_id: str,
    filename: str,
    document_type: str,
    report: ReportResponse,
) -> None:
    """
    Luu ket qua single-doc upload vao Supabase.
    Loi Supabase khong crash request — chi log warning.
    """
    try:
        extracted = report.extracted
        score = report.score

        row = {
            "id": report_id,
            "filename": filename,
            "document_type": document_type,
            "lane": score.lane.value,
            "score": float(score.score),
            "total_co2e_kg": float(score.total_co2e_kg),
            "transport_co2e_kg": float(score.transport_co2e_kg),
            "packaging_co2e_kg": float(score.packaging_co2e_kg),
            "origin_port": extracted.origin_port.value,
            "destination_port": extracted.destination_port.value,
            "transport_mode": extracted.transport_mode.value,
            "distance_km": float(extracted.distance_km.value) if extracted.distance_km.value is not None else None,
            "cargo_weight_tons": float(extracted.cargo_weight_tons.value) if extracted.cargo_weight_tons.value is not None else None,
            "explanation": report.explanation,
            "flags": report.flags or [],
            "needs_human_review": bool(report.needs_human_review),
            "emission_factors": dict(score.emission_factors_used),
            "upload_mode": "single",
            "conflict_count": 0,
            "critical_conflict_count": 0,
            "files_uploaded": 1,
            "document_types_found": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        _get_supabase().table(settings.supabase_table_reports).insert(row).execute()
        log.info(f"Supabase: logged single report {report_id}")

    except Exception as e:
        log.warning(f"Supabase logging that bai (single): {e}")


def _log_multi_to_supabase(
    report_id: str,
    filenames: list[str],
    report: MultiDocumentReportResponse,
) -> None:
    """
    Luu ket qua multi-doc upload vao Supabase.
    Luu them cac truong: conflict_count, critical_conflict_count, files_uploaded.
    """
    try:
        extracted = report.merged
        score = report.score  # None khi halted_for_review

        critical_count = sum(
            1 for c in report.conflicts
            if c.severity.value == "critical"
        )

        row = {
            "id": report_id,
            "filename": " | ".join(filenames),
            "document_type": "multi_document",
            "lane": score.lane.value if score else "HALTED",
            "score": float(score.score) if score else None,
            "total_co2e_kg": float(score.total_co2e_kg) if score else None,
            "transport_co2e_kg": float(score.transport_co2e_kg) if score else None,
            "packaging_co2e_kg": float(score.packaging_co2e_kg) if score else None,
            "origin_port": extracted.origin_port.value,
            "destination_port": extracted.destination_port.value,
            "transport_mode": extracted.transport_mode.value,
            "distance_km": float(extracted.distance_km.value) if extracted.distance_km.value is not None else None,
            "cargo_weight_tons": float(extracted.cargo_weight_tons.value) if extracted.cargo_weight_tons.value is not None else None,
            "explanation": report.explanation,
            "flags": report.flags or [],
            "needs_human_review": bool(report.needs_human_review),
            "emission_factors": dict(score.emission_factors_used) if score else {},
            "upload_mode": "multi",
            "conflict_count": int(len(report.conflicts)),
            "critical_conflict_count": int(critical_count),
            "files_uploaded": int(len(filenames)),
            "document_types_found": [dt.value for dt in report.document_types_found],
            "needs_human_review": bool(report.halted_for_review),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        _get_supabase().table(settings.supabase_table_reports).insert(row).execute()
        log.info(f"Supabase: logged multi report {report_id}")

    except Exception as e:
        log.warning(f"Supabase logging that bai (multi): {e}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_file(file: UploadFile, file_bytes: bytes) -> None:
    """Validate file size va content type. Raise HTTPException neu invalid."""
    if len(file_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File '{file.filename}' qua lon "
                f"({len(file_bytes) // 1024 // 1024:.1f}MB). "
                f"Gioi han {_MAX_FILE_SIZE // 1024 // 1024}MB."
            ),
        )
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"File '{file.filename}': content type '{file.content_type}' "
                "khong duoc ho tro. Can PDF hoac anh (JPEG/PNG/WEBP)."
            ),
        )


# ---------------------------------------------------------------------------
# Single document route (backward compatible)
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=ReportResponse)
def upload_document(
    file: Annotated[UploadFile, File(...)],
    document_type: str = Form(default="invoice"),
) -> ReportResponse:
    """Upload mot chung tu don le va tinh ESG score."""
    file_bytes = file.file.read()
    _validate_file(file, file_bytes)

    filename = file.filename or "document"
    log.info(f"Upload (single): {filename} ({file.content_type}, {len(file_bytes)} bytes)")

    try:
        state = run_pipeline(
            file_bytes=file_bytes,
            filename=filename,
            document_type=document_type,
        )
    except Exception as e:
        log.exception(f"Pipeline loi: {e}")
        raise HTTPException(status_code=500, detail="Xu ly tai lieu that bai.")

    extracted = state.get("extracted")
    esg_score = state.get("esg_score")
    vessel_eff = state.get("vessel_efficiency")
    explanation = state.get("explanation", "")

    if extracted is None or esg_score is None:
        raise HTTPException(
            status_code=422,
            detail=state.get("error") or "Khong the extract thong tin tu tai lieu.",
        )

    report = ReportResponse(
        extracted=extracted,
        score=esg_score,
        vessel_efficiency=vessel_eff,
        explanation=explanation,
    )

    report_id = str(uuid.uuid4())
    _log_single_to_supabase(report_id, filename, document_type, report)

    return report


# ---------------------------------------------------------------------------
# Multi-document route
# ---------------------------------------------------------------------------

async def _process_single_file(
    file_bytes: bytes,
    filename: str,
) -> FileAnalysis:
    """
    Xu ly mot file trong batch: goi extract_and_classify() trong thread pool
    de khong block event loop (Groq API call la blocking I/O).

    Loi duoc catch per-file — mot file loi khong crash toan bo batch.
    """
    loop = asyncio.get_event_loop()
    try:
        classification, extracted = await loop.run_in_executor(
            None,  # dung default ThreadPoolExecutor
            extract_and_classify,
            file_bytes,
            filename,
        )
        return FileAnalysis(
            filename=filename,
            doc_type=classification.doc_type,
            classification_confidence=classification.confidence,
            classification_method=classification.method,
            extracted=extracted,
        )
    except Exception as e:
        log.exception(f"Xu ly file that bai: {filename} — {e}")
        return FileAnalysis(
            filename=filename,
            doc_type=DocumentType.UNKNOWN,
            classification_confidence=0.0,
            classification_method="error",
            extracted=None,
            error=str(e),
        )


@router.post("/upload/multi", response_model=MultiDocumentReportResponse)
async def upload_multi_document(
    files: Annotated[list[UploadFile], File(...)],
) -> MultiDocumentReportResponse:
    """
    Upload nhieu chung tu cung luc (CI + PL + BL + optional TDS/PPWR).

    Pipeline:
    1. Validate tung file (size + content type)
    2. Extract + classify song song (asyncio.gather)
    3. Merge cac extracted documents, detect conflict
    4. Tinh ESG score tren merged document
    5. Generate explanation bang LLM
    6. Tra ve MultiDocumentReportResponse

    Mot file loi khong crash toan bo batch:
    - FileAnalysis.error se co gia tri
    - Cac file hop le van duoc xu ly tiep
    """
    if not files:
        raise HTTPException(status_code=422, detail="Can upload it nhat 1 file.")

    if len(files) > _MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"Toi da {_MAX_FILES_PER_BATCH} file moi lan upload.",
        )

    # Doc va validate tat ca files truoc khi bat dau xu ly
    validated: list[tuple[bytes, str]] = []
    for f in files:
        file_bytes = await f.read()
        _validate_file(f, file_bytes)
        validated.append((file_bytes, f.filename or "document"))

    log.info(f"Multi upload: {len(validated)} files — {[fn for _, fn in validated]}")

    # Xu ly song song
    tasks = [
        _process_single_file(file_bytes, filename)
        for file_bytes, filename in validated
    ]
    file_analyses: list[FileAnalysis] = await asyncio.gather(*tasks)

    # Kiem tra co it nhat 1 file xu ly thanh cong
    successful = [fa for fa in file_analyses if fa.extracted is not None]
    if not successful:
        failed_reasons = [fa.error or "unknown error" for fa in file_analyses]
        raise HTTPException(
            status_code=422,
            detail=f"Khong the xu ly bat ky file nao. Loi: {'; '.join(failed_reasons)}",
        )

    # Merge documents va detect conflict
    merged_doc, conflicts = merge_documents(file_analyses)

    # Tong hop document types
    document_types_found = list({fa.doc_type for fa in file_analyses if fa.doc_type != DocumentType.UNKNOWN})

    # Human-in-the-loop: neu co CRITICAL conflict, dung pipeline.
    # Log vao Supabase voi halted_for_review=True de giu audit trail,
    # nhung khong tinh score — data chua du tin cay.
    critical_conflicts = [c for c in conflicts if c.severity.value == "critical"]
    if critical_conflicts:
        log.warning(
            f"CRITICAL conflicts detected ({len(critical_conflicts)} fields) — "
            f"halting pipeline, score not calculated."
        )
        transport_mode_str = str(merged_doc.transport_mode.value or "unknown")
        missing_types = identify_missing_document_types(document_types_found, transport_mode_str)

        report = MultiDocumentReportResponse(
            per_file=file_analyses,
            merged=merged_doc,
            conflicts=conflicts,
            score=None,
            vessel_efficiency=None,
            explanation="",
            flags=merged_doc.low_confidence_fields,
            halted_for_review=True,
            document_types_found=document_types_found,
            missing_recommended_types=missing_types,
        )

        report_id = str(uuid.uuid4())
        filenames = [fn for _, fn in validated]
        _log_multi_to_supabase(report_id, filenames, report)
        return report

    # Khong co CRITICAL conflict — chay pipeline binh thuong
    from backend.core.agent import run_pipeline_from_doc

    try:
        filled_doc, esg_score, vessel_eff, aircraft_result, explanation = run_pipeline_from_doc(merged_doc)
    except Exception as e:
        log.exception(f"Pipeline (from doc) that bai: {e}")
        raise HTTPException(status_code=500, detail=f"Tinh toan ESG score that bai: {e}")

    transport_mode_str = str(filled_doc.transport_mode.value or "unknown")
    missing_types = identify_missing_document_types(document_types_found, transport_mode_str)

    report = MultiDocumentReportResponse(
        per_file=file_analyses,
        merged=filled_doc,
        conflicts=conflicts,
        score=esg_score,
        vessel_efficiency=vessel_eff,
        aircraft_result=aircraft_result,
        explanation=explanation,
        flags=filled_doc.low_confidence_fields,
        halted_for_review=False,
        document_types_found=document_types_found,
        missing_recommended_types=missing_types,
    )

    report_id = str(uuid.uuid4())
    filenames = [fn for _, fn in validated]
    _log_multi_to_supabase(report_id, filenames, report)
    return report