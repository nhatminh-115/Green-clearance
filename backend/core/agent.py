"""
core/agent.py

LangGraph agent orchestrate toan bo pipeline:
    extract -> check_missing -> fill_distance -> calculate -> explain

Agent chi kick in khi co field bi thieu (distance_km = null).
Neu du thong tin thi pipeline chay thang khong qua vong lap.

Public API:
    run_pipeline(file_bytes, filename, document_type)
        -> single-document flow, goi tu /api/v1/upload

    run_pipeline_from_doc(extracted_doc)
        -> multi-document flow, goi tu /api/v1/upload/multi sau khi merge
        -> skip extract node, bat dau tu check_missing

    generate_explanation(doc, score)
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
    explanation: str

    # Control flow
    missing_fields: Annotated[list[str], operator.add]
    retry_count: int
    error: str | None


# ---------------------------------------------------------------------------
# Haversine distance estimation
# ---------------------------------------------------------------------------
#
# Methodology: Haversine (great-circle / "as-the-crow-flies") x 1.20 sea routing factor.
# Factor 1.20 la conservative estimate cho indirect sea routing (straits, traffic lanes).
# Reference: GLEC Framework v3.2 Section 3.2 — default sea routing factor 1.2-1.3.
# Confidence set 0.60 (thap hon direct extract 1.0, cao hon unknown 0.0).
# Flag "distance_estimated" se duoc them vao low_confidence_fields de UI hien warning.
# ---------------------------------------------------------------------------

import math

# Routing correction factors theo transport mode va khoang cach.
#
# SEA: phai di vong qua eo bien (Malacca, Suez, Panama) — factor cao hon nhieu.
#   Calibrated tu actual sea distances vs great-circle (searoutes.com data).
#   Format: (max_straight_km, factor)
_SEA_ROUTING_FACTORS: list[tuple[float, float]] = [
    (2000,  1.20),   # Intra-Asia, South China Sea
    (5000,  1.15),   # Intra-Asia extended, Indian Ocean short
    (8000,  1.30),   # Asia <-> Middle East / South Asia (qua Malacca)
    (12000, 1.60),   # Asia <-> Europe (qua Malacca + Suez)
    (float("inf"), 1.05),  # Transpacific / transatlantic (gan thang)
]

# AIR: bay gan great-circle, chi lech nhe do ATC routing va traffic separation.
# DEFRA GHG standard cho international air freight: 1.08 (bao gom ca multi-leg deviation).
# Neu co routing_stops tu AWB: tinh tong tung leg x 1.02 (ATC deviation per leg).
_AIR_ROUTING_FACTOR = 1.08   # DEFRA GHG standard, GLEC v3.2 Section 4.3
_AIR_PER_LEG_FACTOR = 1.02   # ATC deviation per individual leg

# TRUCK: di theo duong bo, detour phu thuoc dia hinh va bien gioi.
# Calibrated tu OpenRouteService data cho cac route chau A pho bien.
_TRUCK_ROUTING_FACTORS: list[tuple[float, float]] = [
    (300,  1.35),   # Short domestic: nhieu duong vong, thanh pho
    (1000, 1.25),   # Medium regional
    (float("inf"), 1.18),  # Long haul cross-border
]

# RAIL: tuong doi thang vi co duong ray co dinh, it vong hon truck.
_RAIL_ROUTING_FACTOR = 1.20


def _get_routing_factor(straight_km: float, transport_mode: str = "sea") -> float:
    """Lay routing factor phu hop voi transport mode va khoang cach."""
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
    # Default: sea
    for max_km, factor in _SEA_ROUTING_FACTORS:
        if straight_km <= max_km:
            return factor
    return 1.05

# Port coordinates (lat, lon) cho cac cang pho bien.
# Source: World Port Index (WPI), NGIA Publication 150.
_PORT_COORDS: dict[str, tuple[float, float]] = {
    # Vietnam
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
    # China
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
    # Southeast Asia
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
    # Korea / Japan
    "busan":                    (35.1028, 129.0403),
    "tokyo":                    (35.6762, 139.6503),
    "yokohama":                 (35.4437, 139.6380),
    "osaka":                    (34.6937, 135.5023),
    "nagoya":                   (35.1815, 136.9066),
    "kobe":                     (34.6913, 135.1956),
    "incheon":                  (37.4563, 126.7052),
    # Middle East
    "dubai":                    (25.2048, 55.2708),
    "jebel ali":                (24.9964, 55.0614),
    "abu dhabi":                (24.4539, 54.3773),
    "colombo":                  (6.9271, 79.8612),
    # South Asia
    "chennai":                  (13.0827, 80.2707),
    "mumbai":                   (19.0760, 72.8777),
    "nhava sheva":              (18.9500, 72.9500),
    "kolkata":                  (22.5726, 88.3639),
    "karachi":                  (24.8607, 67.0011),
    "chittagong":               (22.3569, 91.7832),
    # Europe
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
    # North America
    "los angeles":              (33.7175, -118.2776),
    "long beach":               (33.7701, -118.1937),
    "new york":                 (40.6643, -74.0003),
    "savannah":                 (32.0835, -81.0998),
    "seattle":                  (47.6062, -122.3321),
    "vancouver":                (49.2827, -123.1207),
    "houston":                  (29.7604, -95.3698),
    # Australia / NZ
    "sydney":                   (-33.8688, 151.2093),
    "melbourne":                (-37.8136, 144.9631),
    "brisbane":                 (-27.4698, 153.0251),
    "fremantle":                (-32.0569, 115.7440),
    "auckland":                 (-36.8485, 174.7633),
    # Africa
    "durban":                   (-29.8587, 31.0218),
    "cape town":                (-33.9249, 18.4241),
    "mombasa":                  (-4.0435, 39.6682),
    "lagos":                    (6.4531, 3.3958),

    # Airports — IATA codes va ten pho bien
    # Vietnam
    "sgn":                      (10.8185, 106.6524),
    "tan son nhat":             (10.8185, 106.6524),
    "tan son nhat airport":     (10.8185, 106.6524),
    "han":                      (21.2212, 105.8072),
    "noi bai":                  (21.2212, 105.8072),
    "noi bai airport":          (21.2212, 105.8072),
    "dad":                      (16.0439, 108.1992),
    "da nang airport":          (16.0439, 108.1992),
    # Asia
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
    # Europe
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
    # North America
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
    # Australia
    "syd":                      (-33.9399, 151.1753),
    "sydney airport":           (-33.9399, 151.1753),
    "mel":                      (-37.6690, 144.8410),
    "melbourne airport":        (-37.6690, 144.8410),
}


# ---------------------------------------------------------------------------
# Nominatim geocoding (OpenStreetMap) — fallback khi _PORT_COORDS miss
# ---------------------------------------------------------------------------

def _nominatim_lookup(port_name: str) -> tuple[float, float] | None:
    """
    Tra cuu toa do cua cang/san bay qua OpenStreetMap Nominatim API.

    Chi goi khi _PORT_COORDS khong co entry phu hop.
    Rate limit: 1 request/second — co delay built-in.
    Timeout: 8 giay.

    Returns (lat, lon) hoac None neu khong tim thay.
    """
    try:
        import httpx
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": port_name, "format": "json", "limit": 1}
        headers = {
            "User-Agent": "GreenClearance/1.0 ESG-logistics-tool contact@example.com"
        }
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                log.info(f"Nominatim: '{port_name}' -> ({lat:.4f}, {lon:.4f})")
                return lat, lon
            log.warning(f"Nominatim: khong tim thay '{port_name}'")
            return None
    except Exception as e:
        log.warning(f"Nominatim lookup that bai cho '{port_name}': {e}")
        return None


def _normalize_port_name(raw: str) -> str:
    """
    Normalize ten cang ve lowercase, bo dau phay va quoc gia.
    Vi du: "CAT LAI PORT HOCHIMINH, VIETNAM" -> "cat lai port hochiminh"
    """
    name = raw.lower().strip()
    # Bo phan quoc gia sau dau phay
    if "," in name:
        name = name.split(",")[0].strip()
    # Bo cac suffix pho bien
    for suffix in [" port", " harbor", " harbour", " terminal", " international"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name


def _lookup_coords(port_name: str) -> tuple[float, float] | None:
    """
    Tim toa do cua cang hoac san bay.

    Thu theo thu tu:
    1. Exact match (instant)
    2. Substring match (instant)
    3. Fuzzy match voi rapidfuzz partial_ratio >= 75 (instant, handles typos/abbreviations)
    4. Nominatim API fallback (network, ~1s)
    """
    normalized = _normalize_port_name(port_name)

    # 1. Exact match
    if normalized in _PORT_COORDS:
        return _PORT_COORDS[normalized]

    # 2. Substring match
    for key, coords in _PORT_COORDS.items():
        if key in normalized or normalized in key:
            return coords

    # 3. Fuzzy match — xu ly typo, viet tat, ten dai co quoc gia
    try:
        from rapidfuzz import process, fuzz
        result = process.extractOne(
            normalized,
            _PORT_COORDS.keys(),
            scorer=fuzz.partial_ratio,
            score_cutoff=75,
        )
        if result:
            matched_key, score, _ = result
            coords = _PORT_COORDS[matched_key]
            log.info(
                f"Fuzzy match: '{normalized}' -> '{matched_key}' "
                f"(score={score:.0f}) -> {coords}"
            )
            # Cache ket qua de khong phai fuzzy lai
            _PORT_COORDS[normalized] = coords
            return coords
    except ImportError:
        log.debug("rapidfuzz khong co, skip fuzzy matching")

    # 4. Nominatim geocoding fallback
    log.info(f"_PORT_COORDS miss cho '{normalized}', thu Nominatim...")
    time.sleep(1.1)
    coords = _nominatim_lookup(port_name)
    if coords:
        _PORT_COORDS[normalized] = coords
        log.debug(f"Nominatim cached: '{normalized}' -> {coords}")
    return coords


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Tinh khoang cach great-circle giua 2 diem (km)."""
    R = 6371.0  # Earth radius km
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
    """
    Estimate transport distance:

    1. Neu co routing_stops tu AWB (air multi-leg):
       Tinh tong distance tung leg (stop[i] -> stop[i+1]) x _AIR_PER_LEG_FACTOR.
       Vi du: ["SGN","HKG","FRA"] -> hav(SGN,HKG) + hav(HKG,FRA), moi leg x 1.02.

    2. Direct route (sea/truck/rail/air khong co stops):
       Haversine x routing_factor theo mode va distance.

    Tra ve None neu khong tim thay toa do.
    """
    # --- Multi-leg air routing tu AWB ---
    if routing_stops and len(routing_stops) >= 2 and transport_mode == "air":
        total_km = 0.0
        valid_legs = 0
        for i in range(len(routing_stops) - 1):
            c1 = _lookup_coords(routing_stops[i])
            c2 = _lookup_coords(routing_stops[i + 1])
            if c1 is None or c2 is None:
                log.warning(
                    f"Khong tim thay toa do cho stop: "
                    f"'{routing_stops[i]}' hoac '{routing_stops[i+1]}'"
                )
                continue
            leg_km = _haversine_km(*c1, *c2) * _AIR_PER_LEG_FACTOR
            total_km += leg_km
            valid_legs += 1
            log.debug(
                f"Leg {routing_stops[i]}->{routing_stops[i+1]}: "
                f"{leg_km:.0f} km (x{_AIR_PER_LEG_FACTOR} ATC)"
            )

        if valid_legs > 0:
            result = round(total_km, 1)
            log.info(
                f"Multi-leg air: {' -> '.join(routing_stops)} = "
                f"{result:.0f} km ({valid_legs} legs)"
            )
            return result
        # Fallback ve direct neu tat ca leg deu miss coords

    # --- Direct route ---
    origin_coords = _lookup_coords(origin)
    dest_coords   = _lookup_coords(destination)

    if origin_coords is None:
        log.warning(f"Khong tim thay toa do cho: '{origin}'")
        return None
    if dest_coords is None:
        log.warning(f"Khong tim thay toa do cho: '{destination}'")
        return None

    # Sea: dung searoute-py de tinh actual sea distance (di qua dung eo bien)
    # chinh xac hon Haversine x factor vi no dung graph duong hang hai thuc te.
    if transport_mode == "sea":
        try:
            import searoute as sr
            # searoute nhan [lon, lat] — nguoc voi Haversine (lat, lon)
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
            # Fallback ve Haversine neu searoute loi
            straight_km = _haversine_km(*origin_coords, *dest_coords)
            factor = _get_routing_factor(straight_km, transport_mode)
            estimated_km = round(straight_km * factor, 1)
            log.info(f"Haversine fallback (sea): {origin} -> {destination} = {estimated_km:.0f} km")
            return estimated_km

    # Air / truck / rail: Haversine x mode-specific factor
    straight_km = _haversine_km(*origin_coords, *dest_coords)
    factor = _get_routing_factor(straight_km, transport_mode)
    estimated_km = round(straight_km * factor, 1)
    log.info(
        f"Haversine ({transport_mode}): {origin} -> {destination} = "
        f"{straight_km:.0f} km (straight) x {factor} = {estimated_km:.0f} km"
    )
    return estimated_km


# ---------------------------------------------------------------------------
# Standalone functions (reused by both graph nodes and direct callers)
# ---------------------------------------------------------------------------

def _try_fill_distance(doc: ExtractedDocument) -> ExtractedDocument:
    """
    Estimate distance_km neu dang null/0.

    - Air + co routing_stops tu AWB: tinh tong tung leg x 1.02 (ATC)
    - Air direct (khong co stops): Haversine x 1.08 (DEFRA GHG)
    - Sea: Haversine x distance-based sea factor (1.15-1.60)
    - Truck: Haversine x road detour factor (1.18-1.35)
    - Rail: Haversine x 1.20

    Confidence:
    - Multi-leg tu AWB: 0.75 (actual routing data)
    - Direct estimate: 0.60 (approximation)
    """
    origin      = str(doc.origin_port.value or "").strip()
    destination = str(doc.destination_port.value or "").strip()

    if not origin or not destination:
        log.warning("Khong the estimate distance: thieu origin hoac destination port")
        return doc

    transport_mode = str(doc.transport_mode.value or "sea").lower()
    routing_stops  = getattr(doc, "routing_stops", None) or []

    has_routing = bool(routing_stops and len(routing_stops) >= 2)
    confidence  = 0.75 if has_routing else 0.60

    estimated_km = _estimate_distance(
        origin, destination,
        transport_mode=transport_mode,
        routing_stops=routing_stops if has_routing else None,
    )
    if estimated_km is None:
        log.warning(f"Khong the estimate distance cho '{origin}' -> '{destination}'")
        return doc

    # Rebuild doc de re-trigger Pydantic validator (model_copy skip validators)
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
    )
    source = f"{len(routing_stops)}-leg AWB routing" if has_routing else f"{transport_mode} estimate"
    log.info(f"Distance filled: {estimated_km} km ({origin} -> {destination}) [{source}]")
    return updated_doc


def generate_explanation(doc: ExtractedDocument, score: ESGScore, vessel_eff: Optional["VesselEfficiencyResult"] = None) -> str:
    """
    Generate ESG explanation bang ngon ngu tu nhien.

    Standalone function — duoc goi boi ca node_explain() (single-doc graph)
    va run_pipeline_from_doc() (multi-doc flow).

    Args:
        doc: ExtractedDocument da duoc merge va resolve conflict.
        score: ESGScore da duoc tinh toan.
        vessel_eff: Thong tin hieu nang hang tau (neu la duong bien).

    Returns:
        Explanation string, hoac fallback message neu Groq that bai.
    """
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)

    cargo_tons = float(doc.cargo_weight_tons.value or 0)
    intensity = round(score.total_co2e_kg / cargo_tons, 1) if cargo_tons > 0 else 0
    lane_context = {
        "GREEN":  "GREEN (score >= 70): shipment meets ESG best practice standards.",
        "YELLOW": "YELLOW (40-70): shipment is average, improvement needed.",
        "RED":    "RED (score < 40): shipment significantly exceeds emission thresholds.",
    }.get(score.lane.value, "")

    vessel_context = ""
    if vessel_eff and getattr(vessel_eff, "efficiency_grade", None):
        vessel_context = f"\nAdditional Sea Freight Context:\n- Vessel Efficiency Grade: {vessel_eff.efficiency_grade} (Source: {vessel_eff.grade_source})\n- Match Confidence: {vessel_eff.confidence_level}\n"

    warning_fields = []
    if getattr(doc, "low_confidence_fields", None):
        warning_fields = doc.low_confidence_fields
    warning_context = ""
    if warning_fields:
        warning_context = f"\nData Quality Warning: The following extracted fields had low confidence or were entirely estimated/missing: {', '.join(warning_fields)}.\n"

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
{vessel_context}{warning_context}
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
1. What the score and lane mean for this specific shipment — mention the intensity ({intensity} kg/ton)
   and why it leads to this lane (compare to thresholds if RED or YELLOW).
2. The single biggest emission driver (transport mode + distance, or packaging disposal).
{"3. Specifically comment on the Vessel Efficiency Grade and how the chosen carrier performed." if vessel_context else ""}
{"4. Mention the data quality warning and how providing actual data (instead of estimations) would make the score more precise." if warning_context else "3. One concrete, specific action to reduce emissions for this route/cargo type."}
Be direct and quantitative where possible. Do not use bullet points. Write in English.
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
    """Node 1: Chay extractor doc chung tu."""
    doc = extract_document(
        file_bytes=state["file_bytes"],
        filename=state["filename"],
        document_type=state["document_type"],
    )
    return {"extracted": doc, "missing_fields": []}


def node_check_missing(state: AgentState) -> dict:
    """
    Node 2: Kiem tra xem co field quan trong nao bi thieu khong.
    Chi check distance_km — field duy nhat agent co the tu dong fill.
    Cac field confidence thap van chay duoc, chi bi flag.
    """
    doc = state["extracted"]
    if doc is None:
        return {"missing_fields": ["extracted_document_null"]}

    missing = []
    distance_val = doc.distance_km.value
    if distance_val is None or (isinstance(distance_val, (int, float)) and float(distance_val) <= 0):
        missing.append("distance_km")

    return {"missing_fields": missing}


def node_fill_distance(state: AgentState) -> dict:
    """
    Node 3: Goi SeaRoutes API de dien distance neu bi thieu.
    Delegate sang _try_fill_distance() de logic co the reuse.
    """
    doc = state["extracted"]
    if doc is None:
        return {}

    updated_doc = _try_fill_distance(doc)
    # Neu doc khong thay doi (fill that bai), van return de state nhat quan
    return {"extracted": updated_doc}


def node_lookup_vessel(state: AgentState) -> dict:
    """Node 3.5: Thu lookup vessel efficiency neu mode la sea."""
    doc = state["extracted"]
    if doc is None or str(doc.transport_mode.value).lower() != "sea":
        return {}

    vessel_name_field = getattr(doc, "vessel_name", None)
    carrier_name_field = getattr(doc, "carrier_name", None)
    voyage_no_field = getattr(doc, "voyage_number", None)
    cargo_type_field = getattr(doc, "cargo_type", None)

    v_name = str(vessel_name_field.value).strip() if vessel_name_field and vessel_name_field.value else None
    if not v_name or v_name.lower() in ("none", "null", "unknown", ""):
        return {}

    c_name = str(carrier_name_field.value).strip() if carrier_name_field and carrier_name_field.value else None
    v_no = str(voyage_no_field.value).strip() if voyage_no_field and voyage_no_field.value else None
    c_type = str(cargo_type_field.value).strip() if cargo_type_field and cargo_type_field.value else None

    # Goi vessel lookup logic
    eff_result = lookup_vessel_efficiency(
        vessel_name=v_name,
        carrier_name=c_name if c_name and c_name.lower() not in ("none", "null", "unknown") else None,
        voyage_number=v_no if v_no and v_no.lower() not in ("none", "null", "unknown") else None,
        cargo_type=c_type if c_type and c_type.lower() not in ("none", "null", "unknown") else None,
    )
    return {"vessel_efficiency": eff_result}


def node_calculate(state: AgentState) -> dict:
    """Node 4: Tinh CO2e va ESG score."""
    doc = state["extracted"]
    if doc is None:
        return {"error": "Khong co ExtractedDocument de tinh toan"}

    score = calculate(doc)
    return {"esg_score": score}


def node_explain(state: AgentState) -> dict:
    """Node 5: Generate explanation — delegate sang generate_explanation()."""
    doc = state["extracted"]
    score = state["esg_score"]

    if doc is None or score is None:
        return {"explanation": "Khong the tao giai thich do thieu du lieu."}

    explanation = generate_explanation(doc, score, state.get("vessel_efficiency"))
    return {"explanation": explanation}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_fill_distance(state: AgentState) -> Literal["fill_distance", "calculate"]:
    """
    Neu distance_km bi thieu va co du dieu kien de goi API -> fill_distance.
    Nguoc lai -> calculate (distance = 0, calculator se log warning nhung khong crash).
    """
    missing = state.get("missing_fields", [])
    doc = state.get("extracted")

    if "distance_km" in missing and doc is not None:
        origin = str(doc.origin_port.value or "").strip()
        destination = str(doc.destination_port.value or "").strip()
        # Haversine khong can API key — chi can co port info
        if origin and destination:
            return "fill_distance"

    return "calculate"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("extract",       node_extract)
    graph.add_node("check_missing", node_check_missing)
    graph.add_node("fill_distance", node_fill_distance)
    graph.add_node("lookup_vessel", node_lookup_vessel)
    graph.add_node("calculate",     node_calculate)
    graph.add_node("explain",       node_explain)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "check_missing")

    graph.add_conditional_edges(
        "check_missing",
        should_fill_distance,
        {
            "fill_distance": "fill_distance",
            "calculate":     "lookup_vessel",
        },
    )

    graph.add_edge("fill_distance", "lookup_vessel")
    graph.add_edge("lookup_vessel", "calculate")
    graph.add_edge("calculate",     "explain")
    graph.add_edge("explain",       END)

    return graph


# Compile mot lan khi module load, reuse cho moi request
_compiled_graph = _build_graph().compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    file_bytes: bytes,
    filename: str,
    document_type: str = "invoice",
) -> dict:
    """
    Single-document flow. Goi tu POST /api/v1/upload.
    Chay full graph: extract -> check_missing -> [fill_distance] -> calculate -> explain.
    """
    initial_state: AgentState = {
        "file_bytes": file_bytes,
        "filename": filename,
        "document_type": document_type,
        "extracted": None,
        "esg_score": None,
        "vessel_efficiency": None,
        "explanation": "",
        "missing_fields": [],
        "retry_count": 0,
        "error": None,
    }
    return _compiled_graph.invoke(initial_state)


def run_pipeline_from_doc(
    extracted_doc: ExtractedDocument,
) -> tuple[ExtractedDocument, ESGScore, Optional[VesselEfficiencyResult], str]:
    """
    Multi-document flow. Goi tu POST /api/v1/upload/multi sau khi merge.
    Skip extract node — nhan ExtractedDocument da merge san.

    Flow:
        1. Attempt fill distance_km neu null (Haversine, same logic as graph)
        2. Lookup vessel efficiency (neu sea)
        3. Calculate ESG score
        4. Generate explanation

    Returns:
        (doc_after_fill, ESGScore, VesselEfficiencyResult | None, explanation_str)
        doc_after_fill: ExtractedDocument co the da duoc patch distance_km —
        caller phai dung doc nay thay vi merged_doc goc de UI hien dung distance.

    Raises:
        Exception tu calculator.calculate() neu co loi nghiem trong.
        generate_explanation() co internal try/except — khong raise.
    """
    doc = extracted_doc

    # Fill distance neu can — same logic nhu node_check_missing + node_fill_distance
    distance_val = doc.distance_km.value
    needs_fill = (
        distance_val is None
        or (isinstance(distance_val, (int, float)) and float(distance_val) <= 0)
    )

    if needs_fill:
        origin = str(doc.origin_port.value or "").strip()
        destination = str(doc.destination_port.value or "").strip()
        if origin and destination:
            doc = _try_fill_distance(doc)
        else:
            log.warning(
                "distance_km null va thieu port info — "
                "calculator se tinh voi distance = 0."
            )

    # Lookup vessel efficiency neu transport mode la sea
    vessel_eff = None
    if doc.transport_mode and doc.transport_mode.value and str(doc.transport_mode.value).lower() == "sea":
        from backend.core.vessel_lookup import lookup_vessel_efficiency
        vessel_eff = lookup_vessel_efficiency(
            vessel_name=getattr(doc.vessel_name, "value", None) if getattr(doc, "vessel_name", None) else None,
            carrier_name=getattr(doc.carrier_name, "value", None) if getattr(doc, "carrier_name", None) else None,
            voyage_number=getattr(doc.voyage_number, "value", None) if getattr(doc, "voyage_number", None) else None,
            cargo_type=getattr(doc.cargo_type, "value", None) if getattr(doc, "cargo_type", None) else None,
        )

    score = calculate(doc)
    explanation = generate_explanation(doc, score, vessel_eff)

    return doc, score, vessel_eff, explanation