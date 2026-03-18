"""
core/merger.py

Merge nhieu ExtractedDocument tu cac file khac nhau thanh mot document thong nhat,
dong thoi detect va classify conflict giua cac file.

Merge strategy: Priority-based + Confidence fallback
- Thu tu priority: BL > CI > PL > TDS > PPWR_DOC (DOCUMENT_PRIORITY)
- Neu hai file co cung priority, lay field co confidence cao hon
- Neu chenh lech confidence < 0.05, flag WARNING va lay file priority cao hon
- Conflict severity duoc tinh dua tren muc do chenh lech gia tri thuc te

Packaging merge:
- Khong override — gop (union) packaging_items tu tat ca cac file
- PL la nguon chinh cho packaging (priority 3), nhung CI/BL cung co the co
- Dedup bang (material, disposal_method) — lay weight tu file priority cao nhat
"""

import logging
from typing import Optional

from backend.models.schemas import (
    ConflictSeverity, DisposalMethod, DocumentType, ExtractedDocument,
    FieldConfidence, FieldConflict, FileAnalysis, PackagingItem, DOCUMENT_PRIORITY,
)

log = logging.getLogger(__name__)

# Cac field scalar can merge (packaging xu ly rieng)
_SCALAR_FIELDS = [
    "transport_mode",
    "origin_port",
    "destination_port",
    "distance_km",
    "cargo_weight_tons",
]

# Field-specific priority override.
# Mot so field co nguon chinh xac khac voi thu tu priority chung.
#
# cargo_weight_tons: CI > PL > BL > TDS > PPWR
#   BL ghi gross/tare/net weight khong nhat quan — LLM co the chon bat ky field nao.
#   CI va PL ghi commercial weight (net weight of goods) — day la so dung cho ESG.
#   BL priority bi ha xuong cuoi de tranh override CI/PL voi sai so.
_FIELD_PRIORITY_OVERRIDE: dict[str, dict[DocumentType, int]] = {
    "cargo_weight_tons": {
        # PL la nguon chinh xac nhat cho cargo weight:
        # - PL ghi ro Net Weight (hang hoa thuan tuy, khong tinh bao bi)
        # - CI ghi Quantity (thuong = net weight nhung co the la commercial qty)
        # - BL ghi Gross Weight (cargo + packaging) — KHONG dung cho ESG calculation
        DocumentType.PL:       1,
        DocumentType.CI:       2,
        DocumentType.BL:       3,
        DocumentType.TDS:      4,
        DocumentType.PPWR_DOC: 5,
        DocumentType.UNKNOWN:  99,
    },
}

# Nguong chenh lech de classify severity cho numeric fields
# CRITICAL: > 10% chenh lech so voi gia tri lon hon
# WARNING:  5-10%
# INFO:     < 5%
_NUMERIC_CRITICAL_PCT = 0.10
_NUMERIC_WARNING_PCT  = 0.05


# ---------------------------------------------------------------------------
# Conflict severity
# ---------------------------------------------------------------------------

def _classify_severity(
    field_name: str,
    values: list[tuple[str, Optional[str], float]],  # (filename, value, confidence)
) -> ConflictSeverity:
    """
    Tinh conflict severity dua tren muc do chenh lech giua cac gia tri.

    Numeric fields (distance_km, cargo_weight_tons):
        Tinh % chenh lech so voi gia tri max.
        >= 10%  -> CRITICAL
        >= 5%   -> WARNING
        < 5%    -> INFO

    Non-numeric fields (transport_mode, ports):
        Neu gia tri khac nhau hoan toan -> CRITICAL (cho transport_mode)
        Neu khac format nhung same meaning -> INFO (e.g. "HCMC" vs "Ho Chi Minh City")
        Default -> WARNING
    """
    # Lay cac gia tri khac null de so sanh
    non_null = [(fn, v, c) for fn, v, c in values if v is not None]
    if len(non_null) <= 1:
        return ConflictSeverity.INFO

    numeric_fields = {"distance_km", "cargo_weight_tons"}

    if field_name in numeric_fields:
        try:
            nums = [float(v) for _, v, _ in non_null]
            max_val = max(nums)
            if max_val == 0:
                return ConflictSeverity.INFO
            diff_pct = (max(nums) - min(nums)) / max_val
            if diff_pct >= _NUMERIC_CRITICAL_PCT:
                return ConflictSeverity.CRITICAL
            if diff_pct >= _NUMERIC_WARNING_PCT:
                return ConflictSeverity.WARNING
            return ConflictSeverity.INFO
        except (ValueError, TypeError):
            return ConflictSeverity.WARNING

    if field_name == "transport_mode":
        unique_vals = {v.lower().strip() for _, v, _ in non_null if v}
        if len(unique_vals) > 1:
            return ConflictSeverity.CRITICAL
        return ConflictSeverity.INFO

    # Port fields: check neu la same location khac ten/format
    if field_name in {"origin_port", "destination_port"}:
        unique_vals = {v.lower().strip() for _, v, _ in non_null if v}
        if len(unique_vals) <= 1:
            return ConflictSeverity.INFO

        # Strategy 1: Extract IATA/LOCODE codes (3 capital letters in parentheses
        # hoac standalone) va compare.
        # Vi du: "SGN Tan Son Nhat..." va "SGN" deu co IATA = SGN -> same location.
        import re
        def extract_iata(s: str) -> set[str]:
            # Match: (XXX) hoac standalone 3-letter uppercase word
            codes = set(re.findall(r'([A-Z]{3})', s.upper()))
            return codes

        all_iata: list[set[str]] = [extract_iata(v) for _, v, _ in non_null if v]
        # Neu cac IATA sets co intersection -> same location, chi la khac format
        if len(all_iata) >= 2:
            intersection = all_iata[0]
            for s in all_iata[1:]:
                intersection = intersection & s
            if intersection:
                return ConflictSeverity.INFO

        # Strategy 2: fuzzy token overlap (e.g. "HCMC" vs "Ho Chi Minh City")
        try:
            from rapidfuzz import fuzz
            sorted_vals = sorted(unique_vals)
            # So sanh tung cap, neu bat ky cap nao similar >= 70% -> INFO
            for i in range(len(sorted_vals)):
                for j in range(i + 1, len(sorted_vals)):
                    score = fuzz.partial_ratio(sorted_vals[i], sorted_vals[j])
                    if score >= 70:
                        return ConflictSeverity.INFO
        except ImportError:
            # Fallback: token overlap
            sorted_vals = sorted(unique_vals)
            v1_tokens = set(sorted_vals[0].split())
            v2_tokens = set(sorted_vals[-1].split())
            overlap = len(v1_tokens & v2_tokens)
            total_unique = len(v1_tokens | v2_tokens)
            if total_unique > 0 and overlap / total_unique >= 0.5:
                return ConflictSeverity.INFO

        return ConflictSeverity.WARNING

    return ConflictSeverity.WARNING


# ---------------------------------------------------------------------------
# Field-level merge
# ---------------------------------------------------------------------------

def _merge_field(
    field_name: str,
    sources: list[tuple[DocumentType, str, FieldConfidence]],  # (doc_type, filename, field)
) -> tuple[FieldConfidence, Optional[FieldConflict]]:
    """
    Merge mot scalar field tu nhieu source.

    Returns:
        (resolved_field, conflict_or_None)
        conflict = None neu tat ca source dong nhat hoac chi co 1 source co gia tri.
    """
    # Loc bo null values va "unknown" values.
    # "unknown" nghia la file do khong biet gia tri — khong phai mot gia tri thuc.
    # Neu giu "unknown" lai, no se override gia tri thuc tu file khac co priority thap hon.
    # Vi du: BL=unknown (priority 1) se override CI=sea (priority 2) — sai hoan toan.
    #
    # Logic: uu tien sources co gia tri thuc truoc, fallback sang "unknown" neu
    # tat ca deu unknown (van can biet la "unknown" de flag cho user).
    _UNKNOWN_SENTINEL = {"unknown", "none", ""}

    known = [
        (dt, fn, fc) for dt, fn, fc in sources
        if fc.value is not None
        and str(fc.value).strip().lower() not in _UNKNOWN_SENTINEL
    ]
    unknown_only = [
        (dt, fn, fc) for dt, fn, fc in sources
        if fc.value is not None
        and str(fc.value).strip().lower() in _UNKNOWN_SENTINEL
    ]

    # Dung known sources neu co, nguoc lai fallback sang unknown sources
    valid = known if known else unknown_only

    if not valid:
        # Tat ca deu null — tra ve null field, khong co conflict
        return FieldConfidence(value=None, confidence=0.0), None

    if len(valid) == 1:
        _, _, fc = valid[0]
        return fc, None

    # Sort theo priority roi confidence.
    # Neu field nay co entry trong _FIELD_PRIORITY_OVERRIDE, dung priority rieng.
    # Nguoc lai dung DOCUMENT_PRIORITY chung.
    field_priority_map = _FIELD_PRIORITY_OVERRIDE.get(field_name, DOCUMENT_PRIORITY)
    valid_sorted = sorted(
        valid,
        key=lambda x: (field_priority_map.get(x[0], 99), -x[2].confidence),
    )

    # Check co conflict khong (cac gia tri co khac nhau khong)
    unique_values = {str(fc.value).strip().lower() for _, _, fc in valid}

    if len(unique_values) == 1:
        # Tat ca dong nhat — lay field tu nguon priority cao nhat
        best_dt, best_fn, best_fc = valid_sorted[0]
        return best_fc, None

    # Co conflict — resolve va tao FieldConflict record
    best_dt, best_fn, best_fc = valid_sorted[0]

    # Check neu best vs second best co confidence qua gan nhau (<= 0.05)
    # va second-best document co priority thap hon nhung confidence cao hon nhieu
    second_dt, second_fn, second_fc = valid_sorted[1]
    priority_gap = DOCUMENT_PRIORITY.get(second_dt, 99) - DOCUMENT_PRIORITY.get(best_dt, 99)
    confidence_gap = second_fc.confidence - best_fc.confidence

    resolution_reason: str
    if priority_gap == 0 and confidence_gap > 0.15:
        # Same priority nhung second-best confidence cao hon nhieu -> dung confidence
        resolved_dt, resolved_fn, resolved_fc = valid_sorted[1]
        resolution_reason = (
            f"Same document priority; {second_fn} has significantly higher "
            f"confidence ({second_fc.confidence:.2f} vs {best_fc.confidence:.2f})"
        )
    else:
        resolved_dt, resolved_fn, resolved_fc = valid_sorted[0]
        dt_name = resolved_dt.value.replace("_", " ").upper()
        resolution_reason = f"{dt_name} has highest document priority (priority={DOCUMENT_PRIORITY.get(resolved_dt, 99)})"

    # Tao values_by_source map de hien thi tren UI
    values_by_source = {fn: str(fc.value) for _, fn, fc in valid}

    # Tao conflict severity
    severity = _classify_severity(
        field_name,
        [(fn, str(fc.value), fc.confidence) for _, fn, fc in valid],
    )

    # Neu severity = INFO (chi khac format, same location/value) ->
    # khong tao conflict record de UI khong hien.
    # INFO severity la khi IATA match hoac fuzzy >= 70% — không phai conflict thật.
    if severity == ConflictSeverity.INFO:
        return resolved_fc, None

    conflict = FieldConflict(
        field_name=field_name,
        values_by_source=values_by_source,
        resolved_value=str(resolved_fc.value),
        resolution_source=resolved_fn,
        resolution_reason=resolution_reason,
        severity=severity,
    )

    return resolved_fc, conflict


# Cac doc type duoc phep cung cap packaging data
_PACKAGING_WEIGHT_SOURCES = {DocumentType.PL}
_PACKAGING_DISPOSAL_SOURCES = {DocumentType.PPWR_DOC, DocumentType.TDS, DocumentType.PL}

# Priority cho disposal_method (thap = uu tien cao hon)
_DISPOSAL_PRIORITY: dict[DocumentType, int] = {
    DocumentType.PPWR_DOC: 1,  # Legal declaration -- cao nhat
    DocumentType.TDS:      2,  # Material spec -- second
    DocumentType.PL:       3,  # Packing list -- fallback
}


# ---------------------------------------------------------------------------
# Packaging merge
# ---------------------------------------------------------------------------

def _merge_packaging(
    file_analyses: list[FileAnalysis],
) -> list[PackagingItem]:
    """
    Merge packaging_items theo role-based strategy.

    WEIGHT source:   PL only.
        BL/CI khong duoc phep contribute weight vi chung ghi cargo weight,
        khong phai weight cua vat lieu dong goi. Doc-type-aware prompts
        trong extractor.py da ngan BL/CI extract packaging_items.

    DISPOSAL source: PPWR_DOC > TDS > PL (theo do tin cay phap ly).
        - PPWR_DOC: legal declaration of conformity, cao nhat.
        - TDS: material spec co recycled_content_pct, second.
        - PL: fallback neu ca hai tren khong co.

    Flow:
        1. Lay packaging_items tu PL (weight + material + disposal ban dau).
        2. Build disposal override map tu PPWR_DOC va TDS
           (key: material.value -> disposal_method).
        3. Ap dung override: neu PL.disposal = unknown va co override -> patch.
        4. Tra ve merged list.
    """
    # --- Step 1: Lay items tu PL ---
    pl_items: list[PackagingItem] = []
    for fa in file_analyses:
        if fa.doc_type not in _PACKAGING_WEIGHT_SOURCES:
            continue
        if fa.extracted and fa.extracted.packaging_items:
            pl_items.extend(fa.extracted.packaging_items)

    if not pl_items:
        log.debug("Khong co packaging items tu PL")
        return []

    # --- Step 2: Build disposal override map tu PPWR_DOC va TDS ---
    # Map: material.value -> (DisposalMethod, confidence, DocumentType)
    disposal_overrides: dict[str, tuple] = {}

    for fa in file_analyses:
        if fa.doc_type not in _DISPOSAL_PRIORITY:
            continue
        if not fa.extracted or not fa.extracted.packaging_items:
            continue

        source_priority = _DISPOSAL_PRIORITY.get(fa.doc_type, 99)

        for item in fa.extracted.packaging_items:
            if item.disposal_method == DisposalMethod.UNKNOWN:
                continue
            mat_key = item.material.value
            existing = disposal_overrides.get(mat_key)
            if existing is None:
                disposal_overrides[mat_key] = (item.disposal_method, item.confidence, fa.doc_type)
            else:
                _, _, existing_dt = existing
                existing_priority = _DISPOSAL_PRIORITY.get(existing_dt, 99)
                if source_priority < existing_priority:
                    disposal_overrides[mat_key] = (item.disposal_method, item.confidence, fa.doc_type)

    if disposal_overrides:
        log.debug(f"Disposal overrides tu PPWR/TDS: {list(disposal_overrides.keys())}")

    # --- Step 3: Ap dung disposal overrides len PL items ---
    merged_items: list[PackagingItem] = []
    for item in pl_items:
        mat_key = item.material.value
        override = disposal_overrides.get(mat_key)

        if override is not None and item.disposal_method == DisposalMethod.UNKNOWN:
            new_disposal, override_conf, override_dt = override
            log.info(
                f"packaging {mat_key}: disposal overridden "
                f"unknown -> {new_disposal.value} "
                f"(from {override_dt.value}, conf={override_conf:.2f})"
            )
            patched = item.model_copy(update={
                "disposal_method": new_disposal,
                "confidence": max(item.confidence, override_conf),
            })
            merged_items.append(patched)
        else:
            merged_items.append(item)

    log.debug(
        f"Packaging merge: {len(pl_items)} PL items, "
        f"{len(disposal_overrides)} disposal overrides applied"
    )
    return merged_items



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_documents(
    file_analyses: list[FileAnalysis],
) -> tuple[ExtractedDocument, list[FieldConflict]]:
    """
    Merge nhieu FileAnalysis thanh mot ExtractedDocument thong nhat.

    Args:
        file_analyses: list FileAnalysis sau khi extract_and_classify() tung file.
                       Chi xu ly cac file co extracted != None.

    Returns:
        (merged_document, conflicts)
        - merged_document: ExtractedDocument da resolve conflict
        - conflicts: list FieldConflict chua tat ca conflicts phat hien duoc
    """
    # Chi xu ly file co extracted thanh cong
    valid_analyses = [fa for fa in file_analyses if fa.extracted is not None]

    if not valid_analyses:
        log.warning("Khong co file nao extract thanh cong, tra ve empty document")
        return _empty_document(), []

    if len(valid_analyses) == 1:
        log.debug("Chi co 1 file hop le, skip merge logic")
        return valid_analyses[0].extracted, []  # type: ignore[return-value]

    conflicts: list[FieldConflict] = []
    resolved_fields: dict[str, FieldConfidence] = {}

    for field_name in _SCALAR_FIELDS:
        sources: list[tuple[DocumentType, str, FieldConfidence]] = []
        for fa in valid_analyses:
            if fa.extracted is None:
                continue
            fc = getattr(fa.extracted, field_name)

            # cargo_weight_tons: BL ghi gross weight (cargo + packaging),
            # khong phai net weight — khong the so sanh truc tiep voi CI/PL.
            # Loai BL khoi conflict detection cho field nay de tranh false positive.
            if field_name == "cargo_weight_tons" and fa.doc_type == DocumentType.BL:
                continue

            sources.append((fa.doc_type, fa.filename, fc))

        resolved_fc, conflict = _merge_field(field_name, sources)
        resolved_fields[field_name] = resolved_fc

        if conflict is not None:
            conflicts.append(conflict)
            log.info(
                f"Conflict detected: {field_name} "
                f"({conflict.severity.value}) -> resolved to '{conflict.resolved_value}' "
                f"from {conflict.resolution_source}"
            )

    # Merge packaging
    merged_packaging = _merge_packaging(valid_analyses)

    # Merge routing_stops: lay tu file co doc_type = BL (AWB) neu co,
    # vi chi AWB moi co routing field chinh xac.
    # Neu nhieu BL/AWB co routing khac nhau, lay file co priority cao nhat.
    merged_routing: list[str] = []
    bl_sources = [
        fa for fa in valid_analyses
        if fa.doc_type == DocumentType.BL
        and fa.extracted is not None
        and fa.extracted.routing_stops
    ]
    if bl_sources:
        # Lay AWB co priority cao nhat (DOCUMENT_PRIORITY BL = 1)
        best_bl = min(bl_sources, key=lambda fa: DOCUMENT_PRIORITY.get(fa.doc_type, 99))
        merged_routing = best_bl.extracted.routing_stops if best_bl.extracted else []
        log.debug(f"Routing stops tu {best_bl.filename}: {merged_routing}")

    merged = ExtractedDocument(
        transport_mode=resolved_fields["transport_mode"],
        origin_port=resolved_fields["origin_port"],
        destination_port=resolved_fields["destination_port"],
        distance_km=resolved_fields["distance_km"],
        cargo_weight_tons=resolved_fields["cargo_weight_tons"],
        packaging_items=merged_packaging,
        routing_stops=merged_routing,
    )

    log.info(
        f"Merge complete: {len(valid_analyses)} files, "
        f"{len(conflicts)} conflicts ({sum(1 for c in conflicts if c.severity == ConflictSeverity.CRITICAL)} critical)"
    )

    return merged, conflicts


# Recommended document set theo transport mode.
# Air freight dung AWB thay vi BL — pipeline classify AWB la BL (closest match)
# nen khong can them doc type moi, chi can update hint message tren UI.
# Truck/rail: CMR/CIM duoc classify la BL hoac UNKNOWN — tuong tu.
_RECOMMENDED_BY_MODE: dict[str, list[DocumentType]] = {
    "sea":     [DocumentType.CI, DocumentType.PL, DocumentType.BL],
    "air":     [DocumentType.CI, DocumentType.PL, DocumentType.BL],   # BL = AWB proxy
    "truck":   [DocumentType.CI, DocumentType.PL, DocumentType.BL],   # BL = CMR proxy
    "rail":    [DocumentType.CI, DocumentType.PL, DocumentType.BL],   # BL = CIM proxy
    "unknown": [DocumentType.CI, DocumentType.PL, DocumentType.BL],
}


def identify_missing_document_types(
    found_types: list[DocumentType],
    transport_mode: str = "unknown",
) -> list[DocumentType]:
    """
    Xac dinh cac loai chung tu nen co nhung chua upload,
    dua tren transport mode da detect duoc.

    Sea:   CI + PL + BL
    Air:   CI + PL + BL (AWB duoc classify la BL)
    Truck: CI + PL + BL (CMR duoc classify la BL)

    Returns list DocumentType chua upload, theo thu tu importance.
    """
    recommended = _RECOMMENDED_BY_MODE.get(
        transport_mode.lower(),
        _RECOMMENDED_BY_MODE["unknown"],
    )
    found_set = set(found_types)
    return [dt for dt in recommended if dt not in found_set]


def _empty_document() -> ExtractedDocument:
    null_field = FieldConfidence(value=None, confidence=0.0)
    return ExtractedDocument(
        transport_mode=null_field,
        origin_port=null_field,
        destination_port=null_field,
        distance_km=null_field,
        cargo_weight_tons=null_field,
        packaging_items=[],
    )