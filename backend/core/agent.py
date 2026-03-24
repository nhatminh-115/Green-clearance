"""
core/agent.py

LangGraph agent orchestrate toan bo pipeline:
    extract -> check_missing -> fill_distance -> lookup_vessel -> lookup_aircraft -> calculate -> explain

Agent chi kick in khi co field bi thieu (distance_km = null).
Neu du thong tin thi pipeline chay thang khong qua vong lap.

Public API:
    run_pipeline(file_bytes, filename, document_type)
        -> single-document flow, goi tu /api/v1/upload

    run_pipeline_from_doc(extracted_doc)
        -> multi-document flow, goi tu /api/v1/upload/multi sau khi merge
        -> skip extract node, bat dau tu check_missing

    generate_explanation(doc, score, vessel_eff, aircraft_result)
        -> standalone function, dung chung cho ca single va multi flow
"""

import logging
import time
from typing import TypedDict, Literal, Optional
from typing_extensions import Annotated
import operator

from langgraph.graph import StateGraph, START, END

from backend.config import get_settings
from backend.models.schemas import (
    ExtractedDocument, ESGScore, FieldConfidence, VesselEfficiencyResult
)
from backend.core.calculator import calculate
from backend.core.extractor import extract_document
from backend.core.vessel_lookup import lookup_vessel_efficiency

log = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Input
    file_bytes: bytes
    filename: str
    document_type: str

    # Working state
    extracted: ExtractedDocument | None
    esg_score: ESGScore | None
    vessel_efficiency: VesselEfficiencyResult | None
    aircraft_result: Optional[object]   # AircraftEmissionResult | None — avoid circular import
    explanation: str

    # Control flow
    missing_fields: Annotated[list[str], operator.add]
    retry_count: int
    error: str | None


# ---------------------------------------------------------------------------
# Haversine + distance estimation
# ---------------------------------------------------------------------------

import math

_SEA_ROUTING_FACTORS: list[tuple[float, float]] = [
    (2000,  1.20),
    (5000,  1.15),
    (8000,  1.30),
    (12000, 1.60),
    (float("inf"), 1.05),
]

_AIR_ROUTING_FACTOR   = 1.08
_AIR_PER_LEG_FACTOR   = 1.02
_TRUCK_ROUTING_FACTORS: list[tuple[float, float]] = [
    (300,  1.35),
    (1000, 1.25),
    (float("inf"), 1.18),
]
_RAIL_ROUTING_FACTOR  = 1.20
_NM_TO_KM = 1.852


def _get_routing_factor(straight_km: float, transport_mode: str = "sea") -> float:
    mode = transport_mode.lower()
    if mode == "air":
        return _AIR_ROUTING_FACTOR
    if mode == "truck":
        for max_km, factor in _TRUCK_ROUTING_FACTORS:
            if straight_km <= max_km:
                return factor
        return 1.18
    if mode == "rail":
        return _RAIL_ROUTING_FACTOR
    for max_km, factor in _SEA_ROUTING_FACTORS:
        if straight_km <= max_km:
            return factor
    return 1.05


_PORT_COORDS: dict[str, tuple[float, float]] = {
    "ho chi minh":              (10.7769, 106.7009),
    "hochiminh":                (10.7769, 106.7009),
    "hcmc":                     (10.7769, 106.7009),
    "saigon":                   (10.7769, 106.7009),
    "cat lai":                  (10.7769, 106.7009),
    "tan cang":                 (10.7769, 106.7009),
    "ho chi minh vietnam":      (10.7769, 106.7009),
    "hai phong":                (20.8449, 106.6881),
    "haiphong":                 (20.8449, 106.6881),
    "da nang":                  (16.0544, 108.2022),
    "danang":                   (16.0544, 108.2022),
    "quy nhon":                 (13.7830, 109.2197),
    "can tho":                  (10.0341, 105.7878),
    "cang cat lai":             (10.7769, 106.7009),
    "cang sai gon":             (10.7769, 106.7009),
    "cang hai phong":           (20.8449, 106.6881),
    "cang da nang":             (16.0544, 108.2022),
    "cang quy nhon":            (13.7830, 109.2197),
    "shanghai":                 (31.2304, 121.4737),
    "guangzhou":                (23.1291, 113.2644),
    "shenzhen":                 (22.5431, 114.0579),
    "ningbo":                   (29.8683, 121.5440),
    "tianjin":                  (39.1422, 117.1767),
    "qingdao":                  (36.0671, 120.3826),
    "xiamen":                   (24.4798, 118.0894),
    "hong kong":                (22.3193, 114.1694),
    "hongkong":                 (22.3193, 114.1694),
    "hong kong china":          (22.3193, 114.1694),
    "singapore":                (1.2897, 103.8501),
    "port klang":               (3.0000, 101.3833),
    "klang":                    (3.0000, 101.3833),
    "laem chabang":             (13.0856, 100.8800),
    "bangkok":                  (13.7563, 100.5018),
    "jakarta":                  (-6.1051, 106.8451),
    "tanjung priok":            (-6.1051, 106.8451),
    "manila":                   (14.5958, 120.9772),
    "surabaya":                 (-7.2492, 112.7508),
    "penang":                   (5.4141, 100.3288),
    "busan":                    (35.1028, 129.0403),
    "tokyo":                    (35.6762, 139.6503),
    "yokohama":                 (35.4437, 139.6380),
    "osaka":                    (34.6937, 135.5023),
    "nagoya":                   (35.1815, 136.9066),
    "kobe":                     (34.6913, 135.1956),
    "incheon":                  (37.4563, 126.7052),
    "dubai":                    (25.2048, 55.2708),
    "jebel ali":                (24.9964, 55.0614),
    "abu dhabi":                (24.4539, 54.3773),
    "colombo":                  (6.9271, 79.8612),
    "chennai":                  (13.0827, 80.2707),
    "mumbai":                   (19.0760, 72.8777),
    "nhava sheva":              (18.9500, 72.9500),
    "kolkata":                  (22.5726, 88.3639),
    "karachi":                  (24.8607, 67.0011),
    "chittagong":               (22.3569, 91.7832),
    "rotterdam":                (51.9244, 4.4777),
    "hamburg":                  (53.5753, 10.0153),
    "antwerp":                  (51.2213, 4.4051),
    "felixstowe":               (51.9539, 1.3518),
    "bremen":                   (53.0793, 8.8017),
    "le havre":                 (49.4944, 0.1079),
    "barcelona":                (41.3851, 2.1734),
    "valencia":                 (39.4699, -0.3763),
    "piraeus":                  (37.9454, 23.6466),
    "istanbul":                 (41.0082, 28.9784),
    "genoa":                    (44.4056, 8.9463),
    "los angeles":              (33.7175, -118.2776),
    "long beach":               (33.7701, -118.1937),
    "new york":                 (40.6643, -74.0003),
    "savannah":                 (32.0835, -81.0998),
    "seattle":                  (47.6062, -122.3321),
    "vancouver":                (49.2827, -123.1207),
    "houston":                  (29.7604, -95.3698),
    "sydney":                   (-33.8688, 151.2093),
    "melbourne":                (-37.8136, 144.9631),
    "brisbane":                 (-27.4698, 153.0251),
    "fremantle":                (-32.0569, 115.7440),
    "auckland":                 (-36.8485, 174.7633),
    "durban":                   (-29.8587, 31.0218),
    "cape town":                (-33.9249, 18.4241),
    "mombasa":                  (-4.0435, 39.6682),
    "lagos":                    (6.4531, 3.3958),
    "sgn":                      (10.8185, 106.6524),
    "tan son nhat":             (10.8185, 106.6524),
    "tan son nhat airport":     (10.8185, 106.6524),
    "han":                      (21.2212, 105.8072),
    "noi bai":                  (21.2212, 105.8072),
    "noi bai airport":          (21.2212, 105.8072),
    "dad":                      (16.0439, 108.1992),
    "da nang airport":          (16.0439, 108.1992),
    "hkg":                      (22.3080, 113.9185),
    "hong kong airport":        (22.3080, 113.9185),
    "pvg":                      (31.1443, 121.8083),
    "shanghai pudong":          (31.1443, 121.8083),
    "pek":                      (40.0799, 116.6031),
    "beijing capital":          (40.0799, 116.6031),
    "sin":                      (1.3644, 103.9915),
    "changi":                   (1.3644, 103.9915),
    "singapore changi":         (1.3644, 103.9915),
    "icn":                      (37.4602, 126.4407),
    "incheon airport":          (37.4602, 126.4407),
    "nrt":                      (35.7720, 140.3929),
    "narita":                   (35.7720, 140.3929),
    "kix":                      (34.4347, 135.2440),
    "kansai airport":           (34.4347, 135.2440),
    "bkk":                      (13.6811, 100.7472),
    "suvarnabhumi":             (13.6811, 100.7472),
    "cgk":                      (-6.1256, 106.6559),
    "soekarno hatta":           (-6.1256, 106.6559),
    "mnl":                      (14.5086, 121.0194),
    "ninoy aquino":             (14.5086, 121.0194),
    "dxb":                      (25.2532, 55.3657),
    "dubai airport":            (25.2532, 55.3657),
    "bom":                      (19.0896, 72.8656),
    "mumbai airport":           (19.0896, 72.8656),
    "fra":                      (50.0379, 8.5622),
    "frankfurt airport":        (50.0379, 8.5622),
    "lhr":                      (51.4775, -0.4614),
    "heathrow":                 (51.4775, -0.4614),
    "london heathrow":          (51.4775, -0.4614),
    "cdg":                      (49.0097, 2.5479),
    "charles de gaulle":        (49.0097, 2.5479),
    "paris cdg":                (49.0097, 2.5479),
    "ams":                      (52.3086, 4.7639),
    "amsterdam schiphol":       (52.3086, 4.7639),
    "schiphol":                 (52.3086, 4.7639),
    "mad":                      (40.4936, -3.5668),
    "madrid barajas":           (40.4936, -3.5668),
    "muc":                      (48.3538, 11.7861),
    "munich airport":           (48.3538, 11.7861),
    "lax":                      (33.9425, -118.4081),
    "los angeles airport":      (33.9425, -118.4081),
    "jfk":                      (40.6413, -73.7781),
    "john f kennedy":           (40.6413, -73.7781),
    "new york jfk":             (40.6413, -73.7781),
    "ord":                      (41.9742, -87.9073),
    "chicago ohare":            (41.9742, -87.9073),
    "sfo":                      (37.6213, -122.3790),
    "san francisco airport":    (37.6213, -122.3790),
    "yvr":                      (49.1967, -123.1815),
    "vancouver airport":        (49.1967, -123.1815),
    "syd":                      (-33.9399, 151.1753),
    "sydney airport":           (-33.9399, 151.1753),
    "mel":                      (-37.6690, 144.8410),
    "melbourne airport":        (-37.6690, 144.8410),
    "lux":                      (49.6233, 6.2044),
    "luxembourg airport":       (49.6233, 6.2044),
    "luxembourg":               (49.6233, 6.2044),
}


def _nominatim_lookup(port_name: str) -> tuple[float, float] | None:
    try:
        import httpx
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": port_name, "format": "json", "limit": 1}
        headers = {"User-Agent": "GreenClearance/1.0 ESG-logistics-tool contact@example.com"}
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                log.info(f"Nominatim: '{port_name}' -> ({lat:.4f}, {lon:.4f})")
                return lat, lon
            return None
    except Exception as e:
        log.warning(f"Nominatim lookup that bai cho '{port_name}': {e}")
        return None


def _normalize_port_name(raw: str) -> str:
    name = raw.lower().strip()
    if "," in name:
        name = name.split(",")[0].strip()
    for suffix in [" port", " harbor", " harbour", " terminal", " international"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name


def _lookup_coords(port_name: str) -> tuple[float, float] | None:
    normalized = _normalize_port_name(port_name)
    if normalized in _PORT_COORDS:
        return _PORT_COORDS[normalized]
    for key, coords in _PORT_COORDS.items():
        if key in normalized or normalized in key:
            return coords
    try:
        from rapidfuzz import process, fuzz
        result = process.extractOne(
            normalized, _PORT_COORDS.keys(),
            scorer=fuzz.partial_ratio, score_cutoff=75,
        )
        if result:
            matched_key, score, _ = result
            coords = _PORT_COORDS[matched_key]
            log.info(f"Fuzzy match: '{normalized}' -> '{matched_key}' (score={score:.0f})")
            _PORT_COORDS[normalized] = coords
            return coords
    except ImportError:
        pass
    log.info(f"_PORT_COORDS miss cho '{normalized}', thu Nominatim...")
    time.sleep(1.1)
    coords = _nominatim_lookup(port_name)
    if coords:
        _PORT_COORDS[normalized] = coords
    return coords


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _estimate_distance(
    origin: str,
    destination: str,
    transport_mode: str = "sea",
    routing_stops: list[str] | None = None,
) -> float | None:
    if routing_stops and len(routing_stops) >= 2 and transport_mode == "air":
        total_km = 0.0
        valid_legs = 0
        for i in range(len(routing_stops) - 1):
            c1 = _lookup_coords(routing_stops[i])
            c2 = _lookup_coords(routing_stops[i + 1])
            if c1 is None or c2 is None:
                continue
            leg_km = _haversine_km(*c1, *c2) * _AIR_PER_LEG_FACTOR
            total_km += leg_km
            valid_legs += 1
        if valid_legs > 0:
            return round(total_km, 1)

    origin_coords = _lookup_coords(origin)
    dest_coords   = _lookup_coords(destination)
    if origin_coords is None or dest_coords is None:
        log.warning(f"Khong tim thay toa do cho: '{origin}' hoac '{destination}'")
        return None

    if transport_mode == "sea":
        try:
            import searoute as sr
            origin_lonlat = [origin_coords[1], origin_coords[0]]
            dest_lonlat   = [dest_coords[1],   dest_coords[0]]
            route = sr.searoute(origin_lonlat, dest_lonlat, units="km")
            props = dict(route).get("properties", {})
            sea_km = round(props.get("length", 0), 1)
            log.info(
                f"searoute: {origin} -> {destination} = {sea_km:.0f} km "
                f"(vs Haversine {_haversine_km(*origin_coords, *dest_coords):.0f} km straight)"
            )
            return sea_km
        except Exception as e:
            log.warning(f"searoute that bai ({e}), fallback sang Haversine")
            straight_km = _haversine_km(*origin_coords, *dest_coords)
            factor = _get_routing_factor(straight_km, transport_mode)
            return round(straight_km * factor, 1)

    straight_km = _haversine_km(*origin_coords, *dest_coords)
    factor = _get_routing_factor(straight_km, transport_mode)
    return round(straight_km * factor, 1)


def _try_fill_distance(doc: ExtractedDocument) -> ExtractedDocument:
    """
    Estimate distance_km neu dang null/0.

    Confidence theo phuong phap uoc tinh:
    - 0.75: Air multi-leg tu AWB routing stops (actual waypoints)
    - 0.72: Sea via searoute-py (actual sea routing graph)
    - 0.60: Haversine x factor (fallback cho tat ca mode khac)
    """
    origin      = str(doc.origin_port.value or "").strip()
    destination = str(doc.destination_port.value or "").strip()

    if not origin or not destination:
        log.warning("Khong the estimate distance: thieu origin hoac destination port")
        return doc

    transport_mode = str(doc.transport_mode.value or "sea").lower()
    routing_stops  = getattr(doc, "routing_stops", None) or []
    has_routing    = bool(routing_stops and len(routing_stops) >= 2)

    estimated_km = _estimate_distance(
        origin, destination,
        transport_mode=transport_mode,
        routing_stops=routing_stops if has_routing else None,
    )
    if estimated_km is None:
        log.warning(f"Khong the estimate distance cho '{origin}' -> '{destination}'")
        return doc

    if has_routing and transport_mode == "air":
        confidence  = 0.75
        source_note = f"{len(routing_stops)}-leg AWB routing"
    elif transport_mode == "sea":
        try:
            import searoute as sr  # noqa: F401
            confidence  = 0.72
            source_note = "searoute-py actual sea routing"
        except Exception:
            confidence  = 0.60
            source_note = "sea Haversine estimate"
    else:
        confidence  = 0.60
        source_note = f"{transport_mode} Haversine estimate"

    updated_doc = ExtractedDocument(
        transport_mode=doc.transport_mode,
        origin_port=doc.origin_port,
        destination_port=doc.destination_port,
        distance_km=FieldConfidence(value=estimated_km, confidence=confidence),
        cargo_weight_tons=doc.cargo_weight_tons,
        packaging_items=doc.packaging_items,
        raw_text=getattr(doc, "raw_text", None),
        vessel_name=getattr(doc, "vessel_name", None),
        carrier_name=getattr(doc, "carrier_name", None),
        voyage_number=getattr(doc, "voyage_number", None),
        cargo_type=getattr(doc, "cargo_type", None),
        routing_stops=getattr(doc, "routing_stops", []),
        source_document_type=getattr(doc, "source_document_type", None),
    )
    log.info(
        f"Distance filled: {estimated_km} km "
        f"({origin} -> {destination}) [{source_note}, conf={confidence}]"
    )
    return updated_doc


def generate_explanation(
    doc: ExtractedDocument,
    score: ESGScore,
    vessel_eff: Optional[VesselEfficiencyResult] = None,
    aircraft_result=None,  # Optional[AircraftEmissionResult]
) -> str:
    """
    Generate ESG explanation bang ngon ngu tu nhien.
    Standalone function — dung chung cho single-doc va multi-doc flow.
    """
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)

    cargo_tons = float(doc.cargo_weight_tons.value or 0)
    intensity  = round(score.total_co2e_kg / cargo_tons, 1) if cargo_tons > 0 else 0
    lane_context = {
        "GREEN":  "GREEN (score >= 70): shipment meets ESG best practice standards.",
        "YELLOW": "YELLOW (40-70): shipment is average, improvement needed.",
        "RED":    "RED (score < 40): shipment significantly exceeds emission thresholds.",
    }.get(score.lane.value, "")

    vessel_context = ""
    if vessel_eff and getattr(vessel_eff, "efficiency_grade", None):
        vessel_context = (
            f"\nAdditional Sea Freight Context:\n"
            f"- Vessel Efficiency Grade: {vessel_eff.efficiency_grade} "
            f"(Source: {vessel_eff.grade_source})\n"
            f"- Match Confidence: {vessel_eff.confidence_level}\n"
        )

    aircraft_context = ""
    if aircraft_result and getattr(aircraft_result, "confidence", None):
        conf = aircraft_result.confidence
        name = getattr(aircraft_result, "aircraft_name", "Unknown")
        if conf in ("high", "medium"):
            aircraft_context = (
                f"\nAir Freight Context:\n"
                f"- Aircraft type: {name} (confidence: {conf})\n"
                f"- Emission factor source: ICAO CEC Methodology v13.1 (2024)\n"
            )
        elif conf == "low":
            rng = getattr(aircraft_result, "uncertainty_range", None)
            range_str = ""
            if rng:
                p5, p95 = rng
                range_str = f" (uncertainty range: {p5:,.0f}–{p95:,.0f} kg CO2 TTW)"
            aircraft_context = (
                f"\nAir Freight Context:\n"
                f"- Aircraft type inferred (not confirmed): {name}{range_str}\n"
                f"- Source: ICAO CEC Methodology v13.1 with Monte Carlo uncertainty\n"
            )

    warning_context = ""
    if getattr(doc, "low_confidence_fields", None):
        flags = doc.low_confidence_fields
        if flags:
            warning_context = (
                f"\nData Quality Warning: Low confidence or estimated fields: "
                f"{', '.join(flags)}.\n"
            )

    prompt = f"""
You are a sustainability analyst writing a brief ESG report for a logistics shipment.
Write 3-4 sentences in clear, non-technical language. Do NOT use bullet points.

Shipment data:
- Transport mode: {doc.transport_mode.value}
- Origin: {doc.origin_port.value}
- Destination: {doc.destination_port.value}
- Distance: {doc.distance_km.value} km
- Cargo weight: {doc.cargo_weight_tons.value} metric tons
- Packaging items: {[f"{p.weight_tons}t {p.material.value} ({p.disposal_method.value})" for p in doc.packaging_items]}
{vessel_context}{aircraft_context}{warning_context}
ESG Results:
- Transport CO2e: {score.transport_co2e_kg:.1f} kg
- Packaging CO2e: {score.packaging_co2e_kg:.1f} kg
- Total CO2e: {score.total_co2e_kg:.1f} kg
- CO2e intensity: {intensity} kg CO2e per metric ton of cargo
- ESG Score: {score.score:.1f}/100
- Lane: {score.lane.value} — {lane_context}

Scoring methodology (GLEC-aligned):
- Score is based on CO2e intensity (kg CO2e / metric ton cargo), NOT total CO2e.
- Best practice threshold: 200 kg/ton = score 100
- Worst case threshold: 10,000 kg/ton = score 0
- Air freight typically emits 50-80x more per ton-km than sea freight.

Explain:
1. What the score and lane mean for this specific shipment — mention the intensity ({intensity} kg/ton).
2. The single biggest emission driver (transport mode + distance, or packaging disposal).
{"3. Comment on vessel efficiency grade and carrier performance." if vessel_context else ""}
{"3. Comment on aircraft type and emission methodology used." if aircraft_context else ""}
{"4. Note data quality warning and suggest improvements." if warning_context else "3. One concrete action to reduce emissions for this route/cargo type."}
Be direct and quantitative. Do not use bullet points. Write in English.
"""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"generate_explanation that bai: {e}")
        return "Unable to generate explanation automatically."


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def node_extract(state: AgentState) -> dict:
    doc = extract_document(
        file_bytes=state["file_bytes"],
        filename=state["filename"],
        document_type=state["document_type"],
    )
    return {"extracted": doc, "missing_fields": []}


def node_check_missing(state: AgentState) -> dict:
    doc = state["extracted"]
    if doc is None:
        return {"missing_fields": ["extracted_document_null"]}
    missing = []
    distance_val = doc.distance_km.value
    if distance_val is None or (isinstance(distance_val, (int, float)) and float(distance_val) <= 0):
        missing.append("distance_km")
    return {"missing_fields": missing}


def node_fill_distance(state: AgentState) -> dict:
    doc = state["extracted"]
    if doc is None:
        return {}
    return {"extracted": _try_fill_distance(doc)}


def node_lookup_vessel(state: AgentState) -> dict:
    """Lookup vessel efficiency neu mode la sea."""
    doc = state["extracted"]
    if doc is None or str(doc.transport_mode.value).lower() != "sea":
        return {}

    vessel_name_field  = getattr(doc, "vessel_name",  None)
    carrier_name_field = getattr(doc, "carrier_name", None)
    voyage_no_field    = getattr(doc, "voyage_number", None)
    cargo_type_field   = getattr(doc, "cargo_type",   None)

    v_name = str(vessel_name_field.value).strip() if vessel_name_field and vessel_name_field.value else None
    if not v_name or v_name.lower() in ("none", "null", "unknown", ""):
        return {}

    c_name = str(carrier_name_field.value).strip() if carrier_name_field and carrier_name_field.value else None
    v_no   = str(voyage_no_field.value).strip()    if voyage_no_field    and voyage_no_field.value    else None
    c_type = str(cargo_type_field.value).strip()   if cargo_type_field   and cargo_type_field.value   else None

    eff_result = lookup_vessel_efficiency(
        vessel_name=v_name,
        carrier_name=c_name if c_name and c_name.lower() not in ("none", "null", "unknown") else None,
        voyage_number=v_no  if v_no   and v_no.lower()   not in ("none", "null", "unknown") else None,
        cargo_type=c_type   if c_type and c_type.lower() not in ("none", "null", "unknown") else None,
    )
    return {"vessel_efficiency": eff_result}


def node_lookup_aircraft(state: AgentState) -> dict:
    """
    Lookup aircraft-specific emission factor neu mode la air.
    Dung ICAO CEC Methodology v13.1 Appendix C fuel consumption table.
    """
    doc = state["extracted"]
    if doc is None or str(doc.transport_mode.value).lower() != "air":
        return {}

    from backend.core.aircraft_lookup import lookup_air_emission, load_aircraft_data
    load_aircraft_data()

    carrier = None
    carrier_field = getattr(doc, "carrier_name", None)
    if carrier_field and carrier_field.value:
        carrier = str(carrier_field.value).strip()

    # Voyage number thường la flight number tren AWB (vd: "CX880")
    flight_no = None
    voyage_field = getattr(doc, "voyage_number", None)
    if voyage_field and voyage_field.value:
        flight_no = str(voyage_field.value).strip()

    # Dung flight number nhu carrier identifier neu khong co carrier name
    carrier_identifier = carrier or flight_no

    distance_val = doc.distance_km.value
    distance_km  = float(distance_val) if distance_val else 0.0
    cargo_tons   = float(doc.cargo_weight_tons.value) if doc.cargo_weight_tons.value else None

    origin = str(doc.origin_port.value or "").strip()
    dest   = str(doc.destination_port.value or "").strip()

    aircraft_result = lookup_air_emission(
        aircraft_icao=None,
        carrier_name=carrier_identifier,
        distance_km=distance_km,
        cargo_weight_tons=cargo_tons,
        origin=origin,
        destination=dest,
    )
    log.info(
        f"Aircraft lookup: carrier='{carrier_identifier}' dist={distance_km:.0f}km "
        f"-> {aircraft_result.aircraft_name} (conf={aircraft_result.confidence})"
    )
    return {"aircraft_result": aircraft_result}


def node_calculate(state: AgentState) -> dict:
    doc = state["extracted"]
    if doc is None:
        return {"error": "Khong co ExtractedDocument de tinh toan"}
    vessel_eff      = state.get("vessel_efficiency")
    aircraft_result = state.get("aircraft_result")
    score = calculate(doc, vessel_eff, aircraft_result)
    return {"esg_score": score}


def node_explain(state: AgentState) -> dict:
    doc   = state["extracted"]
    score = state["esg_score"]
    if doc is None or score is None:
        return {"explanation": "Khong the tao giai thich do thieu du lieu."}
    explanation = generate_explanation(
        doc, score,
        state.get("vessel_efficiency"),
        state.get("aircraft_result"),
    )
    return {"explanation": explanation}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_fill_distance(state: AgentState) -> Literal["fill_distance", "lookup_vessel"]:
    missing = state.get("missing_fields", [])
    doc     = state.get("extracted")
    if "distance_km" in missing and doc is not None:
        origin      = str(doc.origin_port.value or "").strip()
        destination = str(doc.destination_port.value or "").strip()
        if origin and destination:
            return "fill_distance"
    return "lookup_vessel"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("extract",          node_extract)
    graph.add_node("check_missing",    node_check_missing)
    graph.add_node("fill_distance",    node_fill_distance)
    graph.add_node("lookup_vessel",    node_lookup_vessel)
    graph.add_node("lookup_aircraft",  node_lookup_aircraft)
    graph.add_node("calculate",        node_calculate)
    graph.add_node("explain",          node_explain)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "check_missing")

    graph.add_conditional_edges(
        "check_missing",
        should_fill_distance,
        {
            "fill_distance": "fill_distance",
            "lookup_vessel": "lookup_vessel",
        },
    )

    graph.add_edge("fill_distance",   "lookup_vessel")
    graph.add_edge("lookup_vessel",   "lookup_aircraft")
    graph.add_edge("lookup_aircraft", "calculate")
    graph.add_edge("calculate",       "explain")
    graph.add_edge("explain",         END)

    return graph


_compiled_graph = _build_graph().compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    file_bytes: bytes,
    filename: str,
    document_type: str = "invoice",
) -> dict:
    """Single-document flow. Goi tu POST /api/v1/upload."""
    initial_state: AgentState = {
        "file_bytes":       file_bytes,
        "filename":         filename,
        "document_type":    document_type,
        "extracted":        None,
        "esg_score":        None,
        "vessel_efficiency": None,
        "aircraft_result":  None,
        "explanation":      "",
        "missing_fields":   [],
        "retry_count":      0,
        "error":            None,
    }
    return _compiled_graph.invoke(initial_state)


def run_pipeline_from_doc(
    extracted_doc: ExtractedDocument,
) -> tuple[ExtractedDocument, ESGScore, Optional[VesselEfficiencyResult], str]:
    """
    Multi-document flow. Goi tu POST /api/v1/upload/multi sau khi merge.
    Skip extract node — nhan ExtractedDocument da merge san.

    Flow:
        1. Fill distance_km neu null (Haversine/searoute)
        2. Lookup vessel efficiency (neu sea)
        3. Lookup aircraft emission (neu air)
        4. Calculate ESG score
        5. Generate explanation

    Returns:
        (doc_after_fill, ESGScore, VesselEfficiencyResult | None, explanation_str)
    """
    doc = extracted_doc

    # Fill distance neu can
    distance_val = doc.distance_km.value
    needs_fill   = (
        distance_val is None
        or (isinstance(distance_val, (int, float)) and float(distance_val) <= 0)
    )
    if needs_fill:
        origin      = str(doc.origin_port.value or "").strip()
        destination = str(doc.destination_port.value or "").strip()
        if origin and destination:
            doc = _try_fill_distance(doc)
        else:
            log.warning("distance_km null va thieu port info — calculator se tinh voi distance = 0.")

    transport_mode_str = str(doc.transport_mode.value or "").lower()

    # Lookup vessel efficiency neu sea
    vessel_eff = None
    if transport_mode_str == "sea":
        v_name = getattr(doc.vessel_name, "value", None) if getattr(doc, "vessel_name", None) else None
        if v_name and str(v_name).lower() not in ("none", "null", "unknown", ""):
            vessel_eff = lookup_vessel_efficiency(
                vessel_name=str(v_name),
                carrier_name=getattr(doc.carrier_name, "value", None) if getattr(doc, "carrier_name", None) else None,
                voyage_number=getattr(doc.voyage_number, "value", None) if getattr(doc, "voyage_number", None) else None,
                cargo_type=getattr(doc.cargo_type, "value", None) if getattr(doc, "cargo_type", None) else None,
            )

    # Lookup aircraft emission neu air
    aircraft_result = None
    if transport_mode_str == "air":
        from backend.core.aircraft_lookup import lookup_air_emission, load_aircraft_data
        load_aircraft_data()

        carrier = getattr(doc.carrier_name, "value", None) if getattr(doc, "carrier_name", None) else None
        voyage  = getattr(doc.voyage_number, "value", None) if getattr(doc, "voyage_number", None) else None
        carrier_identifier = str(carrier) if carrier else (str(voyage) if voyage else None)

        dist_val = doc.distance_km.value
        dist_km  = float(dist_val) if dist_val else 0.0
        cargo_kg = float(doc.cargo_weight_tons.value) if doc.cargo_weight_tons.value else None

        aircraft_result = lookup_air_emission(
            aircraft_icao=None,
            carrier_name=carrier_identifier,
            distance_km=dist_km,
            cargo_weight_tons=cargo_kg,
            origin=str(doc.origin_port.value or ""),
            destination=str(doc.destination_port.value or ""),
        )

    score       = calculate(doc, vessel_eff, aircraft_result)
    explanation = generate_explanation(doc, score, vessel_eff, aircraft_result)

    return doc, score, vessel_eff, aircraft_result, explanation