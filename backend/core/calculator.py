"""
core/calculator.py

Tinh toan CO2e va ESG score tu ExtractedDocument.

Methodology:
    - Transport: EPA GHG Hub 2025 Table 8 (TTW) x WtW correction ratio
      WtW correction derived from GLEC Framework v3.2 Module 1, Table p.84
      Source: IMO MEPC 81 marine fuel emission factors
    - Packaging: EPA GHG Hub 2025 Table 9 (disposal emissions)
    - Scoring: CO2e intensity (kg CO2e / metric ton cargo), GLEC-aligned
"""
import logging
from backend.config import get_settings
from backend.models.schemas import (
    CalculationStep, SubStep, ExtractedDocument, ESGScore, ESGLane,
    TransportMode, PackagingMaterial, DisposalMethod,
)
from backend.core.rag import query_all_factors, TransportFactor, PackagingFactor

log = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------

_SHORT_TON_TO_METRIC_TON = 0.907185
_MILE_TO_KM = 1.60934

# EPA factor unit: kg CO2 / short ton-mile -> kg CO2 / metric ton-km
_EPA_TRANSPORT_CONVERSION = _SHORT_TON_TO_METRIC_TON * _MILE_TO_KM  # ~1.4598

# EPA Table 9 unit: metric ton CO2e / short ton material
# -> metric ton CO2e / metric ton material
_EPA_PACKAGING_CONVERSION = _SHORT_TON_TO_METRIC_TON


# ---------------------------------------------------------------------------
# Well-to-Wheel (WtW) correction factors
#
# EPA SmartWay emission factors (Table 8) chi tinh Tank-to-Wheel (TTW):
# chi CO2 phat sinh khi dot nhien lieu trong dong co.
#
# GLEC Framework v3.2 (ISO 14083) yeu cau tinh Well-to-Wheel (WtW):
# bao gom ca CO2 phat sinh tu khai thac, loc, van chuyen nhien lieu
# den truoc khi vao dong co (Well-to-Tank / WTT).
#
# Ratio WtW/TTW duoc tinh tu GLEC Framework v3.2 Module 1:
#   - Sea (VLSFO/HFO): Table p.84, IMO MEPC 81
#     HFO(VLSFO): TTW=3.16, WTW=3.84 kg CO2e/kg -> ratio = 3.84/3.16 = 1.215
#   - Air (Jet A): Table p.79, GREET 2023 / North American sources
#     Jet A: TTW=3.04, WTW=3.74 kg CO2e/kg -> ratio = 3.74/3.04 = 1.230
#     (dung Gasoline ratio lam proxy vi Jet A khong list rieng trang 79,
#      aviation fuel similar carbon intensity to gasoline)
#   - Truck (Diesel): Table p.79, GREET 2023
#     Diesel: TTW=3.22, WTW=3.87 kg CO2e/kg -> ratio = 3.87/3.22 = 1.202
#   - Rail (Diesel): cung nguon Diesel tu Table p.79
#     Assume diesel traction (phu hop VN/Asia context)
#     ratio = 1.202
#
# Assumption duoc ghi ro:
#   - Sea: assume VLSFO (IMO 2020 sulphur cap, post-2020 fleet)
#   - Air: proxy tu gasoline ratio, deviation < 2% so voi Jet A actual
#   - Rail: assume diesel traction (VN/Asia default)
#   - Fuel-specific data requires carrier primary disclosure (GLEC Primary data)
# ---------------------------------------------------------------------------

_WTW_CORRECTION: dict[TransportMode, float] = {
    TransportMode.SEA:   1.215,  # HFO/VLSFO, GLEC v3.2 p.84, IMO MEPC 81
    TransportMode.AIR:   1.230,  # Jet A proxy, GLEC v3.2 p.79, GREET 2023
    TransportMode.TRUCK: 1.202,  # Diesel, GLEC v3.2 p.79, GREET 2023
    TransportMode.RAIL:  1.202,  # Diesel traction, GLEC v3.2 p.79, GREET 2023
    TransportMode.UNKNOWN: 1.202,  # Conservative default: diesel
}

# Fuel type assumption per mode (for explanation generation)
_FUEL_ASSUMPTION: dict[TransportMode, str] = {
    TransportMode.SEA:   "VLSFO (assumed, IMO 2020 sulphur cap)",
    TransportMode.AIR:   "Jet A kerosene",
    TransportMode.TRUCK: "Diesel",
    TransportMode.RAIL:  "Diesel (assumed, VN/Asia context)",
    TransportMode.UNKNOWN: "Diesel (default)",
}


# ---------------------------------------------------------------------------
# ESG scoring thresholds
# ---------------------------------------------------------------------------

_SCORE_GREEN_THRESHOLD  = 70.0
_SCORE_YELLOW_THRESHOLD = 50.0
_INTENSITY_BEST  = 200.0    # kg CO2e / metric ton -> score 100
_INTENSITY_WORST = 10000.0  # kg CO2e / metric ton -> score 0


# ---------------------------------------------------------------------------
# Transport CO2e
# ---------------------------------------------------------------------------

def _calc_transport_co2e(
    doc: ExtractedDocument,
    factor: TransportFactor,
) -> tuple[float, dict[str, float], list[CalculationStep]]:
    cargo_tons   = float(doc.cargo_weight_tons.value or 0.0)
    distance_km  = float(doc.distance_km.value or 0.0)

    if cargo_tons <= 0 or distance_km <= 0:
        return 0.0, {}, []

    # Step 1: Convert EPA factor tu short ton-mile sang metric ton-km (TTW)
    factor_ttw_per_ton_km = factor.co2_per_ton_mile / _EPA_TRANSPORT_CONVERSION

    # Step 2: Apply WtW correction de ra WtW factor
    mode = factor.mode
    wtw_ratio = _WTW_CORRECTION.get(mode, 1.202)
    factor_wtw_per_ton_km = factor_ttw_per_ton_km * wtw_ratio

    # Step 3: Tinh CO2e WtW
    co2e_ttw_kg = cargo_tons * distance_km * factor_ttw_per_ton_km
    co2e_wtw_kg = cargo_tons * distance_km * factor_wtw_per_ton_km

    origin = doc.origin_port.value or "?"
    dest   = doc.destination_port.value or "?"
    fuel   = _FUEL_ASSUMPTION.get(mode, "Diesel")

    transport_sub_steps = [
        SubStep(
            label="Step 1 — Unit conversion: EPA factor -> metric ton-km (TTW)",
            formula=(
                f"{round(factor.co2_per_ton_mile, 6)} kg CO2/short ton-mile"
                f" ÷ {_EPA_TRANSPORT_CONVERSION:.4f}"
            ),
            result=f"= {round(factor_ttw_per_ton_km, 6)} kg CO2 / metric ton-km (TTW)",
            note=(
                f"Conversion: 1 short ton = {_SHORT_TON_TO_METRIC_TON} metric ton, "
                f"1 mile = {_MILE_TO_KM} km → factor = {_SHORT_TON_TO_METRIC_TON} × {_MILE_TO_KM} "
                f"= {_EPA_TRANSPORT_CONVERSION:.4f} | Source: {factor.source}"
            ),
        ),
        SubStep(
            label="Step 2 — WtW correction: Tank-to-Wheel → Well-to-Wheel",
            formula=(
                f"{round(factor_ttw_per_ton_km, 6)}"
                f" × {wtw_ratio} (WtW/TTW ratio)"
            ),
            result=f"= {round(factor_wtw_per_ton_km, 6)} kg CO2e / metric ton-km (WtW)",
            note=(
                f"Fuel assumed: {fuel} | "
                f"WtW/TTW ratio from GLEC Framework v3.2 Module 1 p.84 (IMO MEPC 81) | "
                f"Ratio = WtW emission factor ÷ TTW emission factor per kg fuel"
            ),
        ),
        SubStep(
            label="Step 3 — CO2e calculation",
            formula=(
                f"{cargo_tons} metric tons"
                f" × {distance_km:,.1f} km"
                f" × {round(factor_wtw_per_ton_km, 6)} kg CO2e/metric ton-km"
            ),
            result=f"= {round(co2e_wtw_kg, 2):,.2f} kg CO2e (WtW)",
            note="",
        ),
    ]

    steps = [CalculationStep(
        label=f"{mode.value.capitalize()} freight — {origin} → {dest}",
        factor_key=f"{mode.value}_kg_co2e_per_metric_ton_km_wtw",
        factor_value=round(factor_wtw_per_ton_km, 6),
        factor_unit="kg CO2e / metric ton-km (WtW)",
        quantity=cargo_tons,
        quantity_unit="metric tons cargo",
        distance_km=distance_km,
        co2e_kg=round(co2e_wtw_kg, 2),
        source=(
            f"{factor.source} | WtW ratio {wtw_ratio} from GLEC Framework v3.2 "
            f"Module 1 p.84 (IMO MEPC 81) | Fuel assumed: {fuel}"
        ),
        sub_steps=transport_sub_steps,
    )]

    emission_factors = {
        f"transport_{mode.value}_ttw_kg_co2_per_metric_ton_km": round(factor_ttw_per_ton_km, 6),
        f"transport_{mode.value}_wtw_kg_co2e_per_metric_ton_km": round(factor_wtw_per_ton_km, 6),
        f"transport_{mode.value}_wtw_ratio": wtw_ratio,
    }
    return co2e_wtw_kg, emission_factors, steps


# ---------------------------------------------------------------------------
# Packaging CO2e
# ---------------------------------------------------------------------------

def _calc_packaging_co2e(
    packaging_factors: list[PackagingFactor],
    packaging_weights: dict[tuple[PackagingMaterial, DisposalMethod], float],
) -> tuple[float, dict[str, float], list[CalculationStep]]:
    total_co2e_kg = 0.0
    emission_factors: dict[str, float] = {}
    steps: list[CalculationStep] = []

    for pf in packaging_factors:
        key = (pf.material, pf.disposal)
        weight_tons = packaging_weights.get(key, 0.0)
        if weight_tons <= 0:
            continue

        factor_per_metric_ton = pf.co2e_per_ton / _EPA_PACKAGING_CONVERSION
        co2e_kg = weight_tons * factor_per_metric_ton * 1000
        total_co2e_kg += co2e_kg

        ef_key = f"packaging_{pf.material.value}_{pf.disposal.value}_mt_co2e_per_mt"
        emission_factors[ef_key] = round(factor_per_metric_ton, 6)

        packaging_sub_steps = [
            SubStep(
                label="Step 1 — Unit conversion: EPA factor -> metric ton basis",
                formula=(
                    f"{pf.co2e_per_ton} metric ton CO2e / short ton"
                    f" ÷ {_EPA_PACKAGING_CONVERSION} (short ton → metric ton)"
                ),
                result=f"= {round(factor_per_metric_ton, 6)} metric ton CO2e / metric ton material",
                note=(
                    f"1 short ton = {_SHORT_TON_TO_METRIC_TON} metric ton | "
                    f"Source: {pf.source}"
                ),
            ),
            SubStep(
                label="Step 2 — CO2e calculation",
                formula=(
                    f"{weight_tons} metric tons"
                    f" × {round(factor_per_metric_ton, 6)} metric ton CO2e/metric ton"
                    f" × 1000 (convert to kg)"
                ),
                result=f"= {round(co2e_kg, 2):,.2f} kg CO2e",
                note="",
            ),
        ]

        steps.append(CalculationStep(
            label=f"{pf.material.value.capitalize()} ({pf.disposal.value})",
            factor_key=ef_key,
            factor_value=round(factor_per_metric_ton, 6),
            factor_unit="metric ton CO2e / metric ton material",
            quantity=weight_tons,
            quantity_unit="metric tons",
            distance_km=None,
            co2e_kg=round(co2e_kg, 2),
            source=pf.source,
            sub_steps=packaging_sub_steps,
        ))

    return total_co2e_kg, emission_factors, steps


# ---------------------------------------------------------------------------
# ESG Score
# ---------------------------------------------------------------------------

def _calc_score(
    total_co2e_kg: float,
    cargo_tons: float,
    distance_km: float | None,
    transport_mode_known: bool,
) -> tuple[float, ESGLane]:
    """
    Tinh ESG score tu CO2e intensity, co penalty khi thieu critical fields.

    Penalty logic:
    - distance_km = null va transport_mode != UNKNOWN:
      Transport CO2e tinh ra 0 vi khong co distance, nhung thuc te phat thai co the
      rat cao. Penalize bang cach cap score tai 45.0 (YELLOW) thay vi cho score cao gia.
      Ly do: toan bo ESG score cua chuyen hang khong the la GREEN khi thong tin
      van chuyen chinh yeu con thieu — day la misleading cho nguoi dung va hoi dong.
    - cargo_tons = null: tra ve 0 / RED nhu cu.
    - transport_mode = UNKNOWN: khong penalty rieng vi da co flag tu extractor.
    """
    if cargo_tons <= 0:
        log.warning("Khong co cargo weight, score = 0")
        return 0.0, ESGLane.RED

    intensity = total_co2e_kg / cargo_tons
    raw   = (_INTENSITY_WORST - intensity) / (_INTENSITY_WORST - _INTENSITY_BEST) * 100
    score = max(0.0, min(100.0, raw))

    # Penalty: distance null + transport_mode known -> CO2e underestimated
    # Cap score tai 45.0 de giua YELLOW, khong cho phep GREEN voi data thieu.
    _MISSING_DISTANCE_SCORE_CAP = 45.0
    if transport_mode_known and (distance_km is None or float(distance_km) <= 0):
        if score > _MISSING_DISTANCE_SCORE_CAP:
            log.warning(
                f"distance_km null voi transport_mode known -> "
                f"cap score {score:.1f} -> {_MISSING_DISTANCE_SCORE_CAP}"
            )
            score = _MISSING_DISTANCE_SCORE_CAP

    if score >= _SCORE_GREEN_THRESHOLD:
        lane = ESGLane.GREEN
    elif score >= _SCORE_YELLOW_THRESHOLD:
        lane = ESGLane.YELLOW
    else:
        lane = ESGLane.RED

    log.debug(f"Intensity: {intensity:.1f} kg/t -> Score: {score:.1f} -> {lane}")
    return round(score, 2), lane


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate(doc: ExtractedDocument) -> ESGScore:
    """
    Entry point chinh. Nhan ExtractedDocument, tra ve ESGScore.

    Transport CO2e duoc tinh theo WtW (Well-to-Wheel) bang cach nhan
    EPA SmartWay TTW factor voi WtW correction ratio tu GLEC Framework v3.2.
    Packaging CO2e dung EPA Table 9 disposal emission factors.
    """
    transport_mode_val = str(doc.transport_mode.value or "unknown")
    try:
        transport_mode = TransportMode(transport_mode_val)
    except ValueError:
        transport_mode = TransportMode.UNKNOWN

    packaging_requests: list[tuple[PackagingMaterial, DisposalMethod]] = []
    packaging_weights:  dict[tuple[PackagingMaterial, DisposalMethod], float] = {}

    for item in doc.packaging_items:
        key = (item.material, item.disposal_method)
        packaging_requests.append(key)
        packaging_weights[key] = packaging_weights.get(key, 0.0) + item.weight_tons

    rag_result = query_all_factors(
        transport_modes=[transport_mode],
        packaging_requests=packaging_requests,
    )

    if rag_result.missing_transport:
        log.warning(f"Khong co transport factor cho: {rag_result.missing_transport}")
    if rag_result.missing_packaging:
        log.warning(f"Khong co packaging factor cho: {rag_result.missing_packaging}")

    all_steps: list[CalculationStep] = []
    all_emission_factors: dict[str, float] = {}
    transport_co2e_kg = 0.0

    if rag_result.transport_factors:
        transport_factor = rag_result.transport_factors[0]
        transport_co2e_kg, transport_ef, transport_steps = _calc_transport_co2e(
            doc, transport_factor
        )
        all_emission_factors.update(transport_ef)
        all_steps.extend(transport_steps)

    packaging_co2e_kg, packaging_ef, packaging_steps = _calc_packaging_co2e(
        rag_result.packaging_factors,
        packaging_weights,
    )
    all_emission_factors.update(packaging_ef)
    all_steps.extend(packaging_steps)

    total_co2e_kg = round(transport_co2e_kg + packaging_co2e_kg, 4)
    cargo_tons    = float(doc.cargo_weight_tons.value or 0.0)

    distance_val = doc.distance_km.value
    distance_km_val = float(distance_val) if distance_val is not None else None
    transport_mode_known = (
        transport_mode != TransportMode.UNKNOWN
        and doc.transport_mode.value is not None
    )

    score, lane = _calc_score(
        total_co2e_kg,
        cargo_tons,
        distance_km=distance_km_val,
        transport_mode_known=transport_mode_known,
    )

    return ESGScore(
        transport_co2e_kg=round(transport_co2e_kg, 4),
        packaging_co2e_kg=round(packaging_co2e_kg, 4),
        total_co2e_kg=total_co2e_kg,
        score=score,
        lane=lane,
        emission_factors_used=all_emission_factors,
        calculation_steps=all_steps,
    )