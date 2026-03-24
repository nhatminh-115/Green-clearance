"""
core/aircraft_lookup.py

Aircraft-type-specific emission factor lookup cho air freight.
Thay the EPA fleet average bang ICAO Carbon Emissions Calculator (CEC)
Methodology v13.1 (August 2024) fuel consumption data.

Methodology:
    Fuel consumption (kg) per flight duoc lay tu ICAO CEC Appendix C,
    theo aircraft equivalent code va distance segment (125nm den 8500nm).
    Linear interpolation giua 2 segment gan nhat cho khoang cach bat ky.

    CO2 per flight = fuel_kg x 3.16  (ICAO standard: 3.16 kg CO2/kg Jet-A)

    Emission factor per ton-km:
        factor = CO2_per_flight / cargo_weight_tons / distance_km

    WtW note:
        ICAO CEC factor la TTW (Tank-to-Wheel) — chi tinh CO2 khi dot nhien lieu.
        GLEC v3.2 yeu cau WtW. Do do can apply WtW ratio 1.230 trong calculator.py,
        giong het cach EPA air factor duoc xu ly truoc day.

Uncertainty handling:
    Neu aircraft type khong xac dinh duoc tu AWB:
    -> Infer candidates tu carrier name + route distance
    -> confidence "low" -> Monte Carlo tren candidates
    -> Tra ve co2e range (p5, mean, p95)

Public API:
    load_aircraft_data() -> None

    lookup_aircraft_factor(
        aircraft_icao_or_equiv, distance_km, cargo_weight_tons
    ) -> AircraftEmissionResult | None

    infer_aircraft_types(
        carrier_name, distance_km
    ) -> list[AircraftCandidate]

    lookup_air_emission(
        aircraft_icao, carrier_name, distance_km,
        cargo_weight_tons, origin, destination
    ) -> AircraftEmissionResult
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FUEL_TABLE_JSON = Path("backend/knowledge_base/raw/icao_fuel_table.json")
_CAPACITY_JSON   = Path("backend/knowledge_base/raw/aircraft_capacity.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NM_TO_KM = 1.852
_ICAO_CO2_FACTOR = 3.16        # kg CO2 per kg Jet-A fuel (ICAO CEC v13.1)
_DEFAULT_LOAD_FACTOR = 0.68    # khi khong co actual cargo weight tu AWB

# EPA fallback khi khong co ICAO data (kg CO2 / metric ton-km, da la WtW)
# Source: EPA GHG Hub 2025 Table 8 + GLEC v3.2 WtW ratio 1.230
_EPA_AIR_FALLBACK_WTW = (1.086 / (0.907185 * 1.60934)) * 1.230

# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class AircraftEmissionResult:
    """
    Ket qua tinh emission factor cho mot aircraft type cu the.

    factor_kg_co2_per_ton_km: TTW factor (kg CO2 / metric ton cargo / km).
        Calculator.py se apply WtW ratio 1.230 len tren con so nay,
        giong het cach no xu ly EPA air factor truoc day.

    co2e_kg: tong CO2 cho shipment (cargo_weight x distance x factor x WtW).
        Duoc tinh trong calculator.py, khong phai o day.

    uncertainty_range: (p5_co2e, p95_co2e) khi confidence = "low".
        Don vi kg CO2 (TTW, truoc WtW correction).
    """
    aircraft_icao: str
    aircraft_name: str
    factor_kg_co2_per_ton_km: float   # TTW
    co2e_kg: float                     # TTW (calculator apply WtW sau)
    confidence: str                    # "high" | "medium" | "low" | "fallback_epa"
    source: str
    distance_km_used: float
    cargo_weight_tons_used: float
    uncertainty_range: Optional[tuple[float, float]] = None
    candidates_used: list[str] = field(default_factory=list)
    aircraft_efficiency_grade: Optional[str] = None   
    percentile_in_type_class: Optional[float] = None 


@dataclass
class AircraftCandidate:
    icao_code: str      # ICAO equivalent code (vd: "77L", "744")
    name: str
    probability: float  # 0.0 - 1.0
    basis: str          # "carrier_fleet" | "distance_range" | "default"


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_fuel_table: dict[str, list[Optional[float]]] = {}   # code -> [fuel_kg x 20]
_distances_nm: list[float] = []
_capacity: dict = {}
_loaded = False


# ---------------------------------------------------------------------------
# Carrier fleet composition
#
# Duoc derive tu public fleet databases (ch-aviation.com, annual reports).
# Keys: IATA 2-letter carrier code (pho bien hon trong flight numbers tren AWB).
# Values: list (ICAO_equiv_code, probability) — phai sum = 1.0.
# ICAO equiv codes tu Appendix B / Appendix C cua ICAO CEC v13.1.
# ---------------------------------------------------------------------------

_CARRIER_FLEET: dict[str, list[tuple[str, float]]] = {
    # FedEx — mixed B747/MD11/B763/A306
    "FX":  [("74H", 0.30), ("74L", 0.25), ("M11", 0.20), ("763", 0.15), ("AB6", 0.10)],
    "FDX": [("74H", 0.30), ("74L", 0.25), ("M11", 0.20), ("763", 0.15), ("AB6", 0.10)],
    # UPS — heavy B747/MD11
    "5X":  [("74H", 0.35), ("74L", 0.25), ("M11", 0.15), ("763", 0.15), ("AB6", 0.10)],
    "UPS": [("74H", 0.35), ("74L", 0.25), ("M11", 0.15), ("763", 0.15), ("AB6", 0.10)],
    # DHL / AeroLogic — B767/A300/B757
    "D0":  [("763", 0.30), ("AB6", 0.25), ("757", 0.20), ("744", 0.15), ("332", 0.10)],
    "DHL": [("763", 0.30), ("AB6", 0.25), ("757", 0.20), ("744", 0.15), ("332", 0.10)],
    # Cargolux — all B747-8F/B747-400F
    "CV":  [("74H", 0.60), ("744", 0.40)],
    "CLX": [("74H", 0.60), ("744", 0.40)],
    # Cathay Pacific Cargo — B747-8F/B777F/B747-400F
    "CX":  [("74H", 0.40), ("77F", 0.35), ("744", 0.25)],
    "CPA": [("74H", 0.40), ("77F", 0.35), ("744", 0.25)],
    # Korean Air Cargo
    "KE":  [("74H", 0.35), ("744", 0.30), ("77L", 0.25), ("332", 0.10)],
    "KAL": [("74H", 0.35), ("744", 0.30), ("77L", 0.25), ("332", 0.10)],
    # Lufthansa Cargo — B777F/MD11
    "LH":  [("77F", 0.50), ("M11", 0.30), ("763", 0.20)],
    "GEC": [("77F", 0.50), ("M11", 0.30), ("763", 0.20)],
    # Air France / KLM Cargo
    "AF":  [("77L", 0.45), ("77W", 0.30), ("332", 0.25)],
    "AFR": [("77L", 0.45), ("77W", 0.30), ("332", 0.25)],
    # Singapore Airlines Cargo
    "SQ":  [("74H", 0.40), ("77L", 0.40), ("744", 0.20)],
    "SIA": [("74H", 0.40), ("77L", 0.40), ("744", 0.20)],
    # Vietnam Airlines (belly cargo)
    "VN":  [("332", 0.40), ("333", 0.30), ("789", 0.30)],
    "HVN": [("332", 0.40), ("333", 0.30), ("789", 0.30)],
    # Emirates SkyCargo
    "EK":  [("77L", 0.50), ("74H", 0.30), ("332", 0.20)],
    "UAE": [("77L", 0.50), ("74H", 0.30), ("332", 0.20)],
    # Turkish Cargo
    "TK":  [("77L", 0.45), ("74H", 0.30), ("332", 0.25)],
    "THY": [("77L", 0.45), ("74H", 0.30), ("332", 0.25)],
    # Qatar Airways Cargo
    "QR":  [("74H", 0.40), ("77L", 0.35), ("332", 0.25)],
    "QTR": [("74H", 0.40), ("77L", 0.35), ("332", 0.25)],
    # China Airlines Cargo
    "CI":  [("74H", 0.50), ("744", 0.30), ("77L", 0.20)],
    "CAL": [("74H", 0.50), ("744", 0.30), ("77L", 0.20)],
    # EVA Air Cargo
    "BR":  [("74H", 0.50), ("77L", 0.30), ("763", 0.20)],
    "EVA": [("74H", 0.50), ("77L", 0.30), ("763", 0.20)],
    # Asiana Cargo
    "OZ":  [("74H", 0.40), ("77F", 0.35), ("763", 0.25)],
    # China Southern Cargo
    "CZ":  [("74H", 0.30), ("333", 0.30), ("763", 0.25), ("32S", 0.15)],
    # China Eastern Cargo
    "MU":  [("332", 0.35), ("763", 0.30), ("74H", 0.20), ("32S", 0.15)],
    # Japan Airlines Cargo
    "JL":  [("77F", 0.50), ("763", 0.30), ("74H", 0.20)],
    "JAL": [("77F", 0.50), ("763", 0.30), ("74H", 0.20)],
    # ANA Cargo
    "NH":  [("77F", 0.45), ("763", 0.35), ("74H", 0.20)],
    # Air China Cargo
    "CA":  [("74H", 0.35), ("77F", 0.30), ("332", 0.20), ("763", 0.15)],
}

# Distance-based fallback khi khong co carrier info
# Tung tuple: (min_km, max_km, [(equiv_code, prob)])
_DISTANCE_DEFAULTS: list[tuple[float, float, list[tuple[str, float]]]] = [
    (0,     2000,  [("763", 0.35), ("AB6", 0.25), ("757", 0.20), ("D10", 0.10), ("332", 0.10)]),
    (2000,  5000,  [("763", 0.30), ("332", 0.30), ("744", 0.20), ("M11", 0.10), ("AB6", 0.10)]),
    (5000,  9000,  [("77L", 0.35), ("74H", 0.30), ("744", 0.20), ("M11", 0.10), ("332", 0.05)]),
    (9000,  99999, [("77L", 0.40), ("74H", 0.40), ("744", 0.20)]),
]

# Capacity table: ICAO equiv code -> (name, max_payload_tons, category)
# Source: manufacturer specs (Boeing, Airbus, McDonnell Douglas)
# Dung de tinh emission per ton khi khong co actual cargo weight tu AWB
_CAPACITY: dict[str, tuple[str, float, str]] = {
    # B747 family
    "741": ("Boeing 747-100F",      90.0,  "wide_heavy"),
    "742": ("Boeing 747-200F",     105.0,  "wide_heavy"),
    "743": ("Boeing 747-300",       70.0,  "wide_medium"),
    "744": ("Boeing 747-400F",     113.0,  "wide_heavy"),
    "74H": ("Boeing 747-8F",       134.2,  "wide_heavy"),
    "74L": ("Boeing 747-400ERF",   113.0,  "wide_heavy"),
    "74C": ("Boeing 747-200F",     105.0,  "wide_heavy"),
    "74E": ("Boeing 747-400",      113.0,  "wide_heavy"),
    "74F": ("Boeing 747-200F adv", 105.0,  "wide_heavy"),
    "74M": ("Boeing 747-400",      113.0,  "wide_heavy"),
    "74N": ("Boeing 747-8F",       134.2,  "wide_heavy"),
    "74R": ("Boeing 747-100",       90.0,  "wide_heavy"),
    "74T": ("Boeing 747-100",       90.0,  "wide_heavy"),
    "74X": ("Boeing 747-200F",     105.0,  "wide_heavy"),
    "74Y": ("Boeing 747-400",      113.0,  "wide_heavy"),
    "74J": ("Boeing 747-400",      113.0,  "wide_heavy"),
    "748": ("Boeing 747-8F",       134.2,  "wide_heavy"),
    # B777 family
    "772": ("Boeing 777-200F",     103.0,  "wide_heavy"),
    "773": ("Boeing 777-300",       70.0,  "wide_medium"),
    "777": ("Boeing 777-200F",     103.0,  "wide_heavy"),
    "77F": ("Boeing 777F",         103.0,  "wide_heavy"),
    "77L": ("Boeing 777-200LRF",   103.0,  "wide_heavy"),
    "77W": ("Boeing 777-300ER",     70.0,  "wide_medium"),
    "77X": ("Boeing 777F",         103.0,  "wide_heavy"),
    # B787 family
    "787": ("Boeing 787-8",         43.0,  "wide_medium"),
    "788": ("Boeing 787-8",         43.0,  "wide_medium"),
    "789": ("Boeing 787-9",         53.0,  "wide_medium"),
    "781": ("Boeing 787-10",        63.0,  "wide_medium"),
    # B767 family
    "762": ("Boeing 767-200F",      45.0,  "wide_medium"),
    "763": ("Boeing 767-300F",      54.9,  "wide_medium"),
    "764": ("Boeing 767-400ER",     60.0,  "wide_medium"),
    "767": ("Boeing 767-300F",      54.9,  "wide_medium"),
    "76F": ("Boeing 767-300F",      54.9,  "wide_medium"),
    "76W": ("Boeing 767-300",       54.9,  "wide_medium"),
    "76X": ("Boeing 767-300F",      54.9,  "wide_medium"),
    "76Y": ("Boeing 767-300F",      54.9,  "wide_medium"),
    # B757 family
    "752": ("Boeing 757-200F",      39.8,  "narrow_medium"),
    "753": ("Boeing 757-300",       46.0,  "narrow_medium"),
    "757": ("Boeing 757-200F",      39.8,  "narrow_medium"),
    "75F": ("Boeing 757-200F",      39.8,  "narrow_medium"),
    "75M": ("Boeing 757-200",       39.8,  "narrow_medium"),
    "75T": ("Boeing 757-200",       39.8,  "narrow_medium"),
    "75W": ("Boeing 757-200",       39.8,  "narrow_medium"),
    # A330/A340 family
    "330": ("Airbus A330-200F",     70.0,  "wide_medium"),
    "332": ("Airbus A330-200F",     70.0,  "wide_medium"),
    "333": ("Airbus A330-300",      54.0,  "wide_medium"),
    "33F": ("Airbus A330-200F",     70.0,  "wide_medium"),
    "33X": ("Airbus A330-300",      54.0,  "wide_medium"),
    "338": ("Airbus A330-800",      60.0,  "wide_medium"),
    "339": ("Airbus A330-900",      60.0,  "wide_medium"),
    "340": ("Airbus A340-200",      38.0,  "wide_medium"),
    "342": ("Airbus A340-200",      38.0,  "wide_medium"),
    "343": ("Airbus A340-300",      42.0,  "wide_medium"),
    "345": ("Airbus A340-500",      58.0,  "wide_medium"),
    "346": ("Airbus A340-600",      65.0,  "wide_medium"),
    # A350 family
    "350": ("Airbus A350-900F",     90.0,  "wide_heavy"),
    "359": ("Airbus A350-900F",     90.0,  "wide_heavy"),
    "351": ("Airbus A350-1000",    109.0,  "wide_heavy"),
    # A380
    "380": ("Airbus A380-800F",    150.0,  "wide_heavy"),
    "388": ("Airbus A380-800F",    150.0,  "wide_heavy"),
    # A300 family
    "310": ("Airbus A310F",         39.0,  "wide_medium"),
    "313": ("Airbus A310F",         39.0,  "wide_medium"),
    "31F": ("Airbus A300F",         53.5,  "wide_medium"),
    "31Y": ("Airbus A300-600F",     53.5,  "wide_medium"),
    "312": ("Airbus A300-600F",     53.5,  "wide_medium"),
    "AB3": ("Airbus A300B4-600F",   53.5,  "wide_medium"),
    "AB4": ("Airbus A300B4-600F",   53.5,  "wide_medium"),
    "AB6": ("Airbus A300B4-600F",   53.5,  "wide_medium"),
    "ABF": ("Airbus A300F",         53.5,  "wide_medium"),
    "ABX": ("Airbus A300B4F",       51.0,  "wide_medium"),
    "ABY": ("Airbus A300B4F",       51.0,  "wide_medium"),
    # MD-11 / DC-10
    "M11": ("McDonnell Douglas MD-11F", 91.0, "wide_heavy"),
    "M1F": ("McDonnell Douglas MD-11F", 91.0, "wide_heavy"),
    "M1M": ("McDonnell Douglas MD-11F", 91.0, "wide_heavy"),
    "D10": ("McDonnell Douglas DC-10F", 77.5, "wide_heavy"),
    "D11": ("McDonnell Douglas DC-10-30F", 77.5, "wide_heavy"),
    "D1C": ("McDonnell Douglas DC-10F", 77.5, "wide_heavy"),
    "D1F": ("McDonnell Douglas DC-10F", 77.5, "wide_heavy"),
    # DC-8
    "DC8": ("Douglas DC-8F",        44.0,  "narrow_medium"),
    "D8F": ("Douglas DC-8F",        44.0,  "narrow_medium"),
    "D8X": ("Douglas DC-8-70F",     44.0,  "narrow_medium"),
    "D8Y": ("Douglas DC-8-60F",     44.0,  "narrow_medium"),
    # IL-76
    "IL9": ("Ilyushin IL-76TD",     50.0,  "wide_medium"),
    # AN-124
    "A4F": ("Antonov AN-124-100",  120.0,  "wide_heavy"),
    # Narrow body
    "738": ("Boeing 737-800BCF",    23.9,  "narrow_small"),
    "737": ("Boeing 737-700",       20.0,  "narrow_small"),
    "736": ("Boeing 737-600",       18.0,  "narrow_small"),
    "735": ("Boeing 737-500",       18.5,  "narrow_small"),
    "734": ("Boeing 737-400",       20.0,  "narrow_small"),
    "733": ("Boeing 737-300",       19.5,  "narrow_small"),
    "732": ("Boeing 737-200F",      17.0,  "narrow_small"),
    "731": ("Boeing 737-100",       15.0,  "narrow_small"),
    "7M8": ("Boeing 737 MAX 8",     22.8,  "narrow_small"),
    "7M9": ("Boeing 737 MAX 9",     25.0,  "narrow_small"),
    "32N": ("Airbus A320neo",       20.0,  "narrow_small"),
    "32Q": ("Airbus A321neo",       28.0,  "narrow_small"),
    "32F": ("Airbus A320F",         20.0,  "narrow_small"),
    "320": ("Airbus A320",          20.0,  "narrow_small"),
    "321": ("Airbus A321",          28.0,  "narrow_small"),
    "319": ("Airbus A319",          16.0,  "narrow_small"),
    "318": ("Airbus A318",          14.0,  "narrow_small"),
    "32A": ("Airbus A320",          20.0,  "narrow_small"),
    "32S": ("Airbus A320S",         20.0,  "narrow_small"),
    "221": ("Airbus A220-100",      15.0,  "narrow_small"),
    "223": ("Airbus A220-300",      18.0,  "narrow_small"),
    "31N": ("Airbus A319neo",       16.0,  "narrow_small"),
}

_LOAD_FACTORS = {
    "wide_heavy":    0.70,
    "wide_medium":   0.68,
    "narrow_medium": 0.65,
    "narrow_small":  0.62,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aircraft_data() -> None:
    """Load ICAO fuel table JSON vao memory. Goi mot lan khi startup."""
    global _fuel_table, _distances_nm, _loaded

    if _loaded:
        return

    if not _FUEL_TABLE_JSON.exists():
        log.warning(f"ICAO fuel table not found at {_FUEL_TABLE_JSON}. Air lookup disabled.")
        return

    with open(_FUEL_TABLE_JSON) as f:
        data = json.load(f)

    _distances_nm = data["_distances_nm"]
    _fuel_table   = data["fuel_table"]
    _loaded = True
    log.info(f"ICAO fuel table loaded: {len(_fuel_table)} aircraft types.")


# ---------------------------------------------------------------------------
# Core interpolation
# ---------------------------------------------------------------------------

def _interpolate_fuel(equiv_code: str, distance_km: float) -> Optional[float]:
    """
    Lay fuel consumption (kg/flight) tai distance_km bang linear interpolation.

    ICAO fuel table dung nautical miles (nm) lam distance unit.
    Bro convert sang km o day: distance_km / 1.852 = distance_nm.

    Neu distance ngoai range:
    - Duoi min (< 125nm = 231km): dung gia tri min.
    - Tren max: extrapolate tuyen tinh tu 2 diem cuoi co data.

    Returns None neu equiv_code khong co trong table.
    """
    vals = _fuel_table.get(equiv_code)
    if vals is None:
        return None

    distance_nm = distance_km / _NM_TO_KM
    distances   = _distances_nm  # [125, 250, 500, ..., 8500]

    # Filter out None values de lay range co data thuc su
    valid_pairs = [
        (d, v) for d, v in zip(distances, vals) if v is not None
    ]
    if not valid_pairs:
        return None

    valid_dists, valid_vals = zip(*valid_pairs)

    # Below range: dung gia tri dau tien
    if distance_nm <= valid_dists[0]:
        return float(valid_vals[0])

    # Above range: extrapolate tuyen tinh tu 2 diem cuoi
    if distance_nm >= valid_dists[-1]:
        if len(valid_dists) >= 2:
            slope = (valid_vals[-1] - valid_vals[-2]) / (valid_dists[-1] - valid_dists[-2])
            return float(valid_vals[-1] + slope * (distance_nm - valid_dists[-1]))
        return float(valid_vals[-1])

    # In-range: linear interpolation
    import numpy as np
    idx = int(np.searchsorted(valid_dists, distance_nm, side="right")) - 1
    d0, d1 = valid_dists[idx], valid_dists[idx + 1]
    v0, v1 = valid_vals[idx],  valid_vals[idx + 1]
    t = (distance_nm - d0) / (d1 - d0)
    return float(v0 + t * (v1 - v0))


def _get_payload_tons(equiv_code: str, cargo_weight_tons: Optional[float]) -> tuple[float, str]:
    """
    Lay actual payload tons de tinh emission per ton.

    Priority:
    1. cargo_weight_tons tu AWB (chinh xac nhat, confidence 1.0)
    2. max_payload x load_factor tu _CAPACITY table (estimate)
    3. Default 25 tons (conservative fallback)
    """
    if cargo_weight_tons and cargo_weight_tons > 0:
        return cargo_weight_tons, "actual AWB cargo weight"

    cap = _CAPACITY.get(equiv_code)
    if cap:
        name, max_payload, category = cap
        lf = _LOAD_FACTORS.get(category, _DEFAULT_LOAD_FACTOR)
        payload = max_payload * lf
        return payload, f"estimated ({max_payload}t max payload x {lf} load factor)"

    return 25.0, "default fallback (25t, unknown aircraft)"


# ---------------------------------------------------------------------------
# Carrier name normalization
# ---------------------------------------------------------------------------

def _extract_carrier_codes(raw: str) -> list[str]:
    """
    Trich xuat IATA carrier codes tu raw carrier name string tu AWB.

    Vi du: "CX012" -> ["CX"]
           "CATHAY PACIFIC CARGO" -> ["CX", "CPA"]
           "LUFTHANSA CARGO AG" -> ["LH", "GEC"]
    """
    if not raw:
        return []

    upper = raw.upper().strip()
    codes = []

    # Flight number pattern: 2-3 letters followed by digits
    m = re.match(r"^([A-Z]{2,3})\d+", upper)
    if m:
        codes.append(m.group(1))

    # Exact carrier code match
    for key in _CARRIER_FLEET:
        if upper.startswith(key) or f" {key} " in f" {upper} ":
            codes.append(key)

    # Keyword matching cho ten hang bay pho bien
    _KEYWORDS: dict[str, list[str]] = {
        "FEDEX":           ["FX", "FDX"],
        "FEDERAL EXPRESS": ["FX", "FDX"],
        "UPS":             ["5X", "UPS"],
        "DHL":             ["D0", "DHL"],
        "CARGOLUX":        ["CV", "CLX"],
        "CATHAY":          ["CX", "CPA"],
        "KOREAN AIR":      ["KE", "KAL"],
        "LUFTHANSA":       ["LH", "GEC"],
        "AIR FRANCE":      ["AF", "AFR"],
        "KLM":             ["KL"],
        "SINGAPORE":       ["SQ", "SIA"],
        "VIETNAM":         ["VN", "HVN"],
        "EMIRATES":        ["EK", "UAE"],
        "TURKISH":         ["TK", "THY"],
        "QATAR":           ["QR", "QTR"],
        "CHINA AIRLINES":  ["CI", "CAL"],
        "EVA":             ["BR", "EVA"],
        "ASIANA":          ["OZ"],
        "CHINA SOUTHERN":  ["CZ"],
        "CHINA EASTERN":   ["MU"],
        "JAL":             ["JL", "JAL"],
        "JAPAN AIRLINES":  ["JL", "JAL"],
        "ANA":             ["NH"],
        "AIR CHINA":       ["CA"],
    }
    for keyword, carrier_codes in _KEYWORDS.items():
        if keyword in upper:
            codes.extend(carrier_codes)

    return list(dict.fromkeys(codes))  # dedup preserve order


# ---------------------------------------------------------------------------
# Aircraft type inference
# ---------------------------------------------------------------------------

def infer_aircraft_types(
    carrier_name: Optional[str],
    distance_km: float,
) -> list[AircraftCandidate]:
    """
    Infer danh sach aircraft type candidates tu carrier va distance.

    Tra ve list sorted by probability descending.
    Neu khong co carrier info, dung distance-based defaults.
    """
    if not _loaded:
        load_aircraft_data()

    raw_candidates: list[tuple[str, float, str]] = []

    # Thu carrier-based inference truoc
    if carrier_name:
        carrier_codes = _extract_carrier_codes(carrier_name)
        for code in carrier_codes:
            fleet = _CARRIER_FLEET.get(code, [])
            if fleet:
                raw_candidates = [(icao, prob, "carrier_fleet") for icao, prob in fleet]
                break

    # Fallback sang distance-based
    if not raw_candidates:
        for min_km, max_km, aircraft_list in _DISTANCE_DEFAULTS:
            if min_km <= distance_km < max_km:
                raw_candidates = [(icao, prob, "distance_range") for icao, prob in aircraft_list]
                break

    # Ultimate fallback
    if not raw_candidates:
        raw_candidates = [
            ("77L", 0.40, "default"),
            ("74H", 0.35, "default"),
            ("744", 0.25, "default"),
        ]

    # Chi giu cac aircraft co data trong fuel table
    valid = [
        (icao, prob, basis) for icao, prob, basis in raw_candidates
        if icao in _fuel_table
    ]
    if not valid:
        valid = raw_candidates  # giu nguyen neu filter loai het

    # Normalize probability
    total = sum(p for _, p, _ in valid)
    if total <= 0:
        total = 1.0

    result = []
    for icao, prob, basis in valid:
        cap = _CAPACITY.get(icao)
        name = cap[0] if cap else icao
        result.append(AircraftCandidate(
            icao_code=icao,
            name=name,
            probability=round(prob / total, 3),
            basis=basis,
        ))

    return sorted(result, key=lambda x: x.probability, reverse=True)


# ---------------------------------------------------------------------------
# Direct lookup
# ---------------------------------------------------------------------------

def lookup_aircraft_factor(
    equiv_code: str,
    distance_km: float,
    cargo_weight_tons: Optional[float] = None,
) -> Optional[AircraftEmissionResult]:
    """
    Lookup emission factor cho mot aircraft type cu the.

    Returns AircraftEmissionResult voi TTW factor, hoac None neu khong co data.
    Calculator.py se apply WtW ratio 1.230 len tren factor nay.
    """
    if not _loaded:
        load_aircraft_data()

    fuel_kg = _interpolate_fuel(equiv_code, distance_km)
    if fuel_kg is None:
        return None

    co2_per_flight = fuel_kg * _ICAO_CO2_FACTOR   # kg CO2 (TTW)
    payload_tons, payload_note = _get_payload_tons(equiv_code, cargo_weight_tons)

    if payload_tons <= 0:
        return None

    # TTW emission factor (kg CO2 / metric ton cargo / km)
    factor_ttw = co2_per_flight / payload_tons / distance_km

    # TTW CO2 cho rieng lo hang nay
    actual_weight = cargo_weight_tons if cargo_weight_tons and cargo_weight_tons > 0 else payload_tons
    co2e_ttw = actual_weight * distance_km * factor_ttw

    cap = _CAPACITY.get(equiv_code)
    aircraft_name = cap[0] if cap else equiv_code

    return AircraftEmissionResult(
        aircraft_icao=equiv_code,
        aircraft_name=aircraft_name,
        factor_kg_co2_per_ton_km=round(factor_ttw, 6),
        co2e_kg=round(co2e_ttw, 2),
        confidence="high",
        source=(
            f"ICAO CEC Methodology v13.1 (Aug 2024) | "
            f"Aircraft: {aircraft_name} | "
            f"Fuel: {fuel_kg:.0f} kg/flight x {_ICAO_CO2_FACTOR} = {co2_per_flight:.0f} kg CO2 | "
            f"Payload: {payload_note} | "
            f"TTW only — WtW ratio 1.230 applied by calculator"
        ),
        distance_km_used=distance_km,
        cargo_weight_tons_used=actual_weight,
    )


# ---------------------------------------------------------------------------
# Uncertainty Monte Carlo
# ---------------------------------------------------------------------------

def _monte_carlo_range(
    candidates: list[AircraftCandidate],
    distance_km: float,
    cargo_weight_tons: Optional[float],
    n: int = 500,
) -> tuple[float, float, float]:
    """
    Monte Carlo simulation tren aircraft type candidates.

    Sample aircraft type theo probability distribution n=500 lan.
    Tra ve (p5, mean, p95) cua CO2 TTW distribution.

    Con so nay la TTW — calculator.py apply WtW 1.230 sau.
    """
    if not candidates:
        return (0.0, 0.0, 0.0)

    codes  = [c.icao_code for c in candidates]
    probs  = [c.probability for c in candidates]
    total  = sum(probs)
    probs  = [p / total for p in probs]

    samples = []
    for _ in range(n):
        chosen = np.random.choice(codes, p=probs)
        result = lookup_aircraft_factor(chosen, distance_km, cargo_weight_tons)
        if result:
            samples.append(result.co2e_kg)
        else:
            w = cargo_weight_tons or 25.0
            samples.append(w * distance_km * (_EPA_AIR_FALLBACK_WTW / 1.230))  # fallback TTW

    arr = np.array(samples)
    return (float(np.percentile(arr, 5)), float(np.mean(arr)), float(np.percentile(arr, 95)))

# Sau khi co result, tinh percentile neu confidence khong phai fallback_epa
def _attach_percentile(result: AircraftEmissionResult, distance_km: float) -> AircraftEmissionResult:
    """Tinh aircraft efficiency grade va percentile, attach vao result."""
    if result.confidence == "fallback_epa":
        return result
    try:
        pct, intensity_ttw, grade = get_aircraft_percentile(
            result.aircraft_icao, distance_km
        )
        result.aircraft_efficiency_grade = grade
        result.percentile_in_type_class  = pct
        log.debug(
            f"Aircraft percentile: {result.aircraft_icao} "
            f"at {distance_km:.0f}km -> grade {grade} (top {pct:.1f}%)"
        )
    except Exception as e:
        log.warning(f"get_aircraft_percentile failed: {e}")
    return result

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def lookup_air_emission(
    aircraft_icao: Optional[str],
    carrier_name: Optional[str],
    distance_km: float,
    cargo_weight_tons: Optional[float],
    origin: Optional[str] = None,
    destination: Optional[str] = None,
) -> AircraftEmissionResult:
    """
    Entry point chinh. Pipeline goi ham nay sau khi extract AWB.

    Logic theo 4 cases:

    Case 1 — Biet exact aircraft ICAO code (tu AWB hoac user input):
        Lookup truc tiep, confidence "high".

    Case 2 — Chi co carrier + route, 1 dominant candidate (prob >= 0.70):
        Dung candidate do, confidence "medium".

    Case 3 — Nhieu candidates (uncertainty mode):
        Monte Carlo tren candidates, tra ve mean + (p5, p95) range.
        confidence "low" — UI se hien warning.

    Case 4 — Khong co data nao ca:
        Fallback EPA fleet average, confidence "fallback_epa".
    """
    if not _loaded:
        load_aircraft_data()

    if not _loaded or distance_km <= 0:
        # Return EPA fallback
        w = cargo_weight_tons or 25.0
        return AircraftEmissionResult(
            aircraft_icao="UNKNOWN",
            aircraft_name="Unknown (EPA fallback)",
            factor_kg_co2_per_ton_km=_EPA_AIR_FALLBACK_WTW,
            co2e_kg=round(w * distance_km * _EPA_AIR_FALLBACK_WTW, 2) if distance_km > 0 else 0.0,
            confidence="fallback_epa",
            source="EPA GHG Hub 2025 Table 8 + GLEC v3.2 WtW 1.230 (fleet average fallback)",
            distance_km_used=distance_km,
            cargo_weight_tons_used=w,
        )

    # Case 1: Biet aircraft type
    if aircraft_icao:
        result = lookup_aircraft_factor(aircraft_icao.upper(), distance_km, cargo_weight_tons)
        if result:
            result.confidence = "high"
            return _attach_percentile(result, distance_km)

    # Case 2 & 3: Infer tu carrier + distance
    candidates = infer_aircraft_types(carrier_name, distance_km)

    if not candidates:
        # No inference possible -> EPA fallback
        w = cargo_weight_tons or 25.0
        log.warning(f"No candidates for carrier='{carrier_name}', falling back to EPA.")
        return AircraftEmissionResult(
            aircraft_icao="UNKNOWN",
            aircraft_name="Unknown (EPA fallback)",
            factor_kg_co2_per_ton_km=_EPA_AIR_FALLBACK_WTW,
            co2e_kg=round(w * distance_km * _EPA_AIR_FALLBACK_WTW, 2),
            confidence="fallback_epa",
            source="EPA GHG Hub 2025 Table 8 + GLEC v3.2 WtW 1.230 (no carrier data)",
            distance_km_used=distance_km,
            cargo_weight_tons_used=w,
        )

    top = candidates[0]

    # Case 2: Single dominant candidate
    if top.probability >= 0.70 and len(candidates) == 1:
        result = lookup_aircraft_factor(top.icao_code, distance_km, cargo_weight_tons)
        if result:
            result.confidence = "medium"
            result.candidates_used = [top.icao_code]
            return _attach_percentile(result, distance_km)

    # Case 3: Multiple candidates -> Monte Carlo
    p5, mean_co2, p95 = _monte_carlo_range(candidates, distance_km, cargo_weight_tons)

    # Dung top candidate lam "representative" cho factor display
    rep = lookup_aircraft_factor(top.icao_code, distance_km, cargo_weight_tons)
    candidate_desc = ", ".join(
        f"{c.icao_code}({c.probability:.0%})" for c in candidates[:4]
    )

    if rep:
        rep.confidence = "low"
        rep.co2e_kg = round(mean_co2, 2)
        rep.uncertainty_range = (round(p5, 2), round(p95, 2))
        rep.candidates_used = [c.icao_code for c in candidates]
        rep.source = (
            f"ICAO CEC Methodology v13.1 (Aug 2024) | "
            f"Aircraft type inferred (not confirmed) | "
            f"Candidates: {candidate_desc} | "
            f"Monte Carlo n=500 — mean used, range (p5={p5:,.0f}, p95={p95:,.0f}) kg CO2 TTW | "
            f"TTW only — WtW ratio 1.230 applied by calculator"
        )
        return _attach_percentile(rep, distance_km)
    
    if result.confidence in ("high", "medium", "low"):
        pct, intensity_ttw, grade = get_aircraft_percentile(
         result.aircraft_icao, distance_km
        )
        result.percentile_in_type_class = pct
        result.aircraft_efficiency_grade = grade

    # EPA fallback neu rep lookup that bai
    w = cargo_weight_tons or 25.0
    return AircraftEmissionResult(
        aircraft_icao="UNKNOWN",
        aircraft_name="Unknown (EPA fallback)",
        factor_kg_co2_per_ton_km=_EPA_AIR_FALLBACK_WTW,
        co2e_kg=round(w * distance_km * _EPA_AIR_FALLBACK_WTW, 2),
        confidence="fallback_epa",
        source="EPA GHG Hub 2025 Table 8 + GLEC v3.2 WtW 1.230 (fallback)",
        distance_km_used=distance_km,
        cargo_weight_tons_used=w,
    )


def get_aircraft_percentile(
    equiv_code: str,
    distance_km: float,
) -> tuple[float, float, str]:
    """
    Tinh percentile cua aircraft type nay so voi cac loai khac
    tren cung distance range trong ICAO CEC table.

    Returns (percentile, intensity_ttw, grade_letter)
    - percentile: 0-100, thap = hieu qua hon (it khi phat thai hon)
    - intensity_ttw: kg CO2 / metric ton-km (TTW) cua aircraft nay
    - grade_letter: A (top 20%) den E (bottom 20%)
    """
    if not _loaded:
        load_aircraft_data()

    # Tinh intensity cho tat ca aircraft types tai distance nay
    intensities: list[float] = []
    own_intensity: float | None = None

    for code, cap_data in _CAPACITY.items():
        fuel_kg = _interpolate_fuel(code, distance_km)
        if fuel_kg is None:
            continue

        name, max_payload, category = cap_data
        lf = _LOAD_FACTORS.get(category, 0.68)
        payload = max_payload * lf
        if payload <= 0:
            continue

        co2_per_flight = fuel_kg * _ICAO_CO2_FACTOR
        intensity = co2_per_flight / payload / distance_km
        intensities.append(intensity)

        if code == equiv_code:
            own_intensity = intensity

    if own_intensity is None or not intensities:
        return (50.0, 0.0, "C")

    # Percentile: bao nhieu % aircraft khac co intensity CAO HON aircraft nay
    # (tuong tu EU MRV: percentile thap = hieu qua hon = tot hon)
    pct = sum(1 for x in intensities if x > own_intensity) / len(intensities) * 100

    # Grade A-E theo quintile
    grade = (
        "A" if pct >= 80 else
        "B" if pct >= 60 else
        "C" if pct >= 40 else
        "D" if pct >= 20 else
        "E"
    )

    return (round(pct, 1), round(own_intensity, 6), grade)