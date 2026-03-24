"""
core/calculator.py

Tinh toan CO2e va ESG score tu ExtractedDocument.

Methodology:
    Transport emissions theo 3 nguon, theo thu tu uu tien:

    1. Sea + EU MRV vessel data (vessel_eff confidence high/medium):
       CO2e = cargo_tons x distance_km x vessel_specific_factor
       vessel_specific_factor derived tu actual reported fuel consumption
       trong EU MRV 2024 dataset. Day la WtW-equivalent (khong apply WtW ratio them).

    2. Air + ICAO CEC aircraft data (aircraft_result confidence high/medium/low):
       CO2e = cargo_tons x distance_km x icao_aircraft_factor x WtW_ratio
       icao_aircraft_factor derived tu ICAO Carbon Emissions Calculator
       Methodology v13.1 (Aug 2024) Appendix C fuel consumption table.
       ICAO data la TTW nen van can apply WtW ratio 1.230.

    3. EPA fleet average fallback (tat ca truong hop con lai):
       CO2e = cargo_tons x distance_km x EPA_factor x WtW_ratio
       EPA GHG Hub 2025 Table 8 + GLEC v3.2 WtW correction.

    Packaging: EPA GHG Hub 2025 Table 9 (disposal emissions).
    Scoring: CO2e intensity (kg CO2e / metric ton cargo), GLEC-aligned.
"""
import logging
from typing import Optional
from backend.config import get_settings
from backend.models.schemas import (
    CalculationStep, SubStep, ExtractedDocument, ESGScore, ESGLane,
    TransportMode, PackagingMaterial, DisposalMethod,
    VesselEfficiencyResult,
)
from backend.core.rag import query_all_factors, TransportFactor, PackagingFactor

log = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------

_SHORT_TON_TO_METRIC_TON = 0.907185
_MILE_TO_KM = 1.60934
_NM_TO_KM = 1.852   # nautical mile -> km (dung cho EU MRV vessel factor)

# EPA factor unit: kg CO2 / short ton-mile -> kg CO2 / metric ton-km
_EPA_TRANSPORT_CONVERSION = _SHORT_TON_TO_METRIC_TON * _MILE_TO_KM  # ~1.4598

# EPA Table 9 unit: metric ton CO2e / short ton material -> / metric ton material
_EPA_PACKAGING_CONVERSION = _SHORT_TON_TO_METRIC_TON

# ICAO CEC: kg CO2 per kg Jet-A fuel (TTW, ICAO standard)
_ICAO_CO2_FACTOR = 3.16


# ---------------------------------------------------------------------------
# Well-to-Wheel (WtW) correction factors
#
# EPA SmartWay (Table 8) va ICAO CEC chi tinh Tank-to-Wheel (TTW):
# chi CO2 phat sinh khi dot nhien lieu trong dong co.
# GLEC Framework v3.2 (ISO 14083) yeu cau Well-to-Wheel (WtW):
# bao gom ca CO2 phat sinh tu khai thac, loc, van chuyen nhien lieu.
#
# Ratio WtW/TTW tu GLEC Framework v3.2 Module 1:
#   Sea  (VLSFO/HFO): HFO TTW=3.16, WTW=3.84 -> ratio = 3.84/3.16 = 1.215
#   Air  (Jet A):     Jet A TTW=3.04, WTW=3.74 -> ratio = 3.74/3.04 = 1.230
#   Truck (Diesel):   Diesel TTW=3.22, WTW=3.87 -> ratio = 3.87/3.22 = 1.202
#   Rail (Diesel):    Same diesel source -> ratio = 1.202
#
# NGOAI LE quan trong:
#   EU MRV vessel data: da la operational emissions (WtW-equivalent)
#   -> KHONG apply WtW ratio, tranh double-counting.
#   ICAO CEC data: la TTW (chi tinh khi dot nhien lieu)
#   -> Van apply WtW ratio 1.230 nhu EPA air factor.
# ---------------------------------------------------------------------------

_WTW_CORRECTION: dict[TransportMode, float] = {
    TransportMode.SEA:     1.215,
    TransportMode.AIR:     1.230,
    TransportMode.TRUCK:   1.202,
    TransportMode.RAIL:    1.202,
    TransportMode.UNKNOWN: 1.202,  # conservative default: diesel
}

_FUEL_ASSUMPTION: dict[TransportMode, str] = {
    TransportMode.SEA:     "VLSFO (assumed, IMO 2020 sulphur cap)",
    TransportMode.AIR:     "Jet A kerosene",
    TransportMode.TRUCK:   "Diesel",
    TransportMode.RAIL:    "Diesel (assumed, VN/Asia context)",
    TransportMode.UNKNOWN: "Diesel (default)",
}


# ---------------------------------------------------------------------------
# ESG scoring thresholds
# ---------------------------------------------------------------------------

_SCORE_GREEN_THRESHOLD  = 70.0
_SCORE_YELLOW_THRESHOLD = 40.0
_INTENSITY_BEST  = 200.0    # kg CO2e / metric ton -> score 100
_INTENSITY_WORST = 10000.0  # kg CO2e / metric ton -> score 0


# ---------------------------------------------------------------------------
# Transport CO2e
# ---------------------------------------------------------------------------

def _calc_transport_co2e(
    doc: ExtractedDocument,
    factor: TransportFactor,
    vessel_eff: Optional[VesselEfficiencyResult] = None,
    aircraft_result=None,   # Optional[AircraftEmissionResult] — avoid circular import
) -> tuple[float, dict[str, float], list[CalculationStep]]:
    """
    Tinh transport CO2e theo 3 nhanh uu tien:
    1. Sea + EU MRV vessel-specific factor (WtW-equivalent, no correction needed)
    2. Air + ICAO CEC aircraft-specific factor (TTW, apply WtW 1.230)
    3. EPA fleet average fallback + WtW correction (tat ca truong hop con lai)
    """
    cargo_tons  = float(doc.cargo_weight_tons.value or 0.0)
    distance_km = float(doc.distance_km.value or 0.0)

    if cargo_tons <= 0 or distance_km <= 0:
        return 0.0, {}, []

    mode   = factor.mode
    origin = doc.origin_port.value or "?"
    dest   = doc.destination_port.value or "?"

    # -----------------------------------------------------------------------
    # NHANH 1: Sea + EU MRV vessel-specific data
    #
    # EU MRV emission_intensity don vi: g CO2 / metric tonne . nautical mile
    # Can convert sang: kg CO2e / metric tonne . km
    #   g -> kg: / 1000
    #   nautical mile -> km: / 1.852
    #
    # EU MRV data la actual fuel consumption (operational, WtW-equivalent)
    # -> KHONG apply WtW ratio, tranh double-counting.
    # -----------------------------------------------------------------------
    use_vessel = (
        mode == TransportMode.SEA
        and vessel_eff is not None
        and vessel_eff.confidence_level in ("high", "medium")
        and vessel_eff.emission_intensity_g_per_tonne_nm is not None
    )

    if use_vessel:
        vessel_factor = vessel_eff.emission_intensity_g_per_tonne_nm / (1000 * _NM_TO_KM)
        co2e_kg = cargo_tons * distance_km * vessel_factor

        sub_steps = [
            SubStep(
                label="Step 1 — Convert EU MRV intensity to metric ton-km basis",
                formula=(
                    f"{vessel_eff.emission_intensity_g_per_tonne_nm:.2f} g CO2/t·nm"
                    f" ÷ 1000 (g→kg) ÷ {_NM_TO_KM} (nm→km)"
                ),
                result=f"= {round(vessel_factor, 6)} kg CO2e / metric ton·km",
                note=(
                    "EU MRV 2024 actual operational emissions — "
                    "WtW correction NOT applied (already WtW-equivalent)"
                ),
            ),
            SubStep(
                label="Step 2 — CO2e calculation",
                formula=(
                    f"{cargo_tons} t × {distance_km:,.1f} km"
                    f" × {round(vessel_factor, 6)} kg CO2e/t·km"
                ),
                result=f"= {round(co2e_kg, 2):,.2f} kg CO2e",
                note="",
            ),
        ]

        steps = [CalculationStep(
            label=f"Sea freight (vessel-specific EU MRV) — {origin} → {dest}",
            factor_key="sea_vessel_specific_kg_co2e_per_metric_ton_km",
            factor_value=round(vessel_factor, 6),
            factor_unit="kg CO2e / metric ton·km (EU MRV actual, WtW-equivalent)",
            quantity=cargo_tons,
            quantity_unit="metric tons cargo",
            distance_km=distance_km,
            co2e_kg=round(co2e_kg, 2),
            source=(
                f"EU MRV 2024 | Vessel: {vessel_eff.vessel_name_matched} | "
                f"Intensity: {vessel_eff.emission_intensity_g_per_tonne_nm:.2f} g/t·nm | "
                f"Match confidence: {vessel_eff.confidence_level} | "
                f"WtW correction NOT applied (EU MRV is operational)"
            ),
            sub_steps=sub_steps,
        )]

        emission_factors = {
            "sea_vessel_specific_g_co2_per_tonne_nm": vessel_eff.emission_intensity_g_per_tonne_nm,
            "sea_vessel_specific_kg_co2e_per_metric_ton_km": round(vessel_factor, 6),
        }
        return co2e_kg, emission_factors, steps

    # -----------------------------------------------------------------------
    # NHANH 2: Air + ICAO CEC aircraft-specific data
    #
    # ICAO CEC Appendix C cho fuel consumption (kg/flight) theo aircraft type
    # va distance segment. Factor duoc tinh:
    #   factor_ttw = CO2_per_flight / cargo_tons / distance_km
    #
    # ICAO CEC la TTW (chi tinh khi dot Jet-A, CO2 factor 3.16 kg/kg fuel)
    # -> Van apply WtW ratio 1.230 nhu EPA air factor.
    #
    # Confidence "low" co uncertainty_range tu Monte Carlo
    # -> Ghi vao source note de breakdown UI hien warning.
    # -----------------------------------------------------------------------
    use_aircraft = (
        mode == TransportMode.AIR
        and aircraft_result is not None
        and aircraft_result.confidence in ("high", "medium", "low")
        and aircraft_result.factor_kg_co2_per_ton_km > 0
    )

    if use_aircraft:
        wtw_ratio    = _WTW_CORRECTION[TransportMode.AIR]  # 1.230
        factor_ttw   = aircraft_result.factor_kg_co2_per_ton_km
        factor_wtw   = factor_ttw * wtw_ratio
        co2e_wtw     = cargo_tons * distance_km * factor_wtw

        # Uncertainty range note cho confidence "low"
        uncertainty_note = ""
        if aircraft_result.confidence == "low" and aircraft_result.uncertainty_range:
            p5, p95 = aircraft_result.uncertainty_range
            # p5/p95 la TTW tu Monte Carlo, convert sang WtW de hien dung don vi
            p5_wtw  = round(p5  * (cargo_tons / aircraft_result.cargo_weight_tons_used) * wtw_ratio, 0)
            p95_wtw = round(p95 * (cargo_tons / aircraft_result.cargo_weight_tons_used) * wtw_ratio, 0)
            uncertainty_note = (
                f" | Uncertainty (p5-p95 WtW): "
                f"{p5_wtw:,.0f} – {p95_wtw:,.0f} kg CO2e "
                f"[aircraft type inferred, not confirmed from AWB]"
            )

        confidence_label = {
            "high":   "aircraft type confirmed from document",
            "medium": "single dominant aircraft type inferred from carrier",
            "low":    "multiple aircraft types possible — Monte Carlo mean used",
        }.get(aircraft_result.confidence, aircraft_result.confidence)

        sub_steps = [
            SubStep(
                label="Step 1 — ICAO CEC Appendix C: fuel burn → CO2 per flight (TTW)",
                formula=(
                    f"fuel_burn(kg) × {_ICAO_CO2_FACTOR} kg CO2/kg Jet-A"
                    f" ÷ {cargo_tons}t ÷ {distance_km:,.0f} km"
                ),
                result=f"= {round(factor_ttw, 6)} kg CO2/metric ton·km (TTW)",
                note=(
                    f"Source: ICAO Carbon Emissions Calculator Methodology v13.1 "
                    f"(Aug 2024) Appendix C | Aircraft: {aircraft_result.aircraft_name} | "
                    f"Confidence: {confidence_label}"
                ),
            ),
            SubStep(
                label="Step 2 — WtW correction (TTW → WtW)",
                formula=f"{round(factor_ttw, 6)} × {wtw_ratio} (WtW/TTW ratio, Jet A)",
                result=f"= {round(factor_wtw, 6)} kg CO2e / metric ton·km (WtW)",
                note=(
                    "GLEC v3.2 Module 1 p.79, GREET 2023 | "
                    "ICAO CEC is TTW-only — WtW correction required "
                    "(unlike EU MRV vessel data which is already WtW-equivalent)"
                ),
            ),
            SubStep(
                label="Step 3 — CO2e calculation",
                formula=(
                    f"{cargo_tons} t × {distance_km:,.1f} km"
                    f" × {round(factor_wtw, 6)} kg CO2e/t·km"
                ),
                result=f"= {round(co2e_wtw, 2):,.2f} kg CO2e (WtW)",
                note="",
            ),
        ]

        steps = [CalculationStep(
            label=f"Air freight (aircraft-specific ICAO CEC) — {origin} → {dest}",
            factor_key="air_aircraft_specific_kg_co2e_per_metric_ton_km_wtw",
            factor_value=round(factor_wtw, 6),
            factor_unit="kg CO2e / metric ton·km (WtW)",
            quantity=cargo_tons,
            quantity_unit="metric tons cargo",
            distance_km=distance_km,
            co2e_kg=round(co2e_wtw, 2),
            source=(
                f"ICAO CEC Methodology v13.1 (Aug 2024) Appendix C | "
                f"Aircraft: {aircraft_result.aircraft_name} | "
                f"WtW ratio {wtw_ratio} (GLEC v3.2 p.79) | "
                f"Confidence: {confidence_label}"
                f"{uncertainty_note}"
            ),
            sub_steps=sub_steps,
        )]

        emission_factors = {
            "air_aircraft_specific_ttw_kg_co2_per_metric_ton_km": round(factor_ttw, 6),
            "air_aircraft_specific_wtw_kg_co2e_per_metric_ton_km": round(factor_wtw, 6),
            "air_wtw_ratio": wtw_ratio,
        }
        return co2e_wtw, emission_factors, steps

    # -----------------------------------------------------------------------
    # NHANH 3: EPA fleet average fallback + WtW correction
    #
    # Chay khi:
    # - Truck hoac Rail (khong co aircraft/vessel dataset)
    # - Sea nhung vessel khong tim thay hoac confidence low
    # - Air nhung khong co aircraft data (carrier unknown, distance out of range)
    # -----------------------------------------------------------------------
    wtw_ratio = _WTW_CORRECTION.get(mode, 1.202)
    fuel      = _FUEL_ASSUMPTION.get(mode, "Diesel")

    factor_ttw_per_ton_km = factor.co2_per_ton_mile / _EPA_TRANSPORT_CONVERSION
    factor_wtw_per_ton_km = factor_ttw_per_ton_km * wtw_ratio
    co2e_wtw_kg = cargo_tons * distance_km * factor_wtw_per_ton_km

    sub_steps = [
        SubStep(
            label="Step 1 — Unit conversion: EPA factor → metric ton-km (TTW)",
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
                f"WtW/TTW ratio from GLEC Framework v3.2 Module 1 p.84 (IMO MEPC 81)"
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
        sub_steps=sub_steps,
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
                label="Step 1 — Unit conversion: EPA factor → metric ton basis",
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

    Penalty 1: distance_km null + transport_mode known
        -> Cap score tai 45.0 (YELLOW) vi transport CO2e bi underestimate nghiem trong.

    Penalty 2: transport_mode = UNKNOWN
        -> Cap score tai 45.0 vi CO2e duoc tinh bang default truck factor,
           khong phan anh dung mode thuc te.
    """
    if cargo_tons <= 0:
        log.warning("Khong co cargo weight, score = 0")
        return 0.0, ESGLane.RED

    intensity = total_co2e_kg / cargo_tons
    raw   = (_INTENSITY_WORST - intensity) / (_INTENSITY_WORST - _INTENSITY_BEST) * 100
    score = max(0.0, min(100.0, raw))

    _MISSING_DISTANCE_SCORE_CAP = 45.0
    if transport_mode_known and (distance_km is None or float(distance_km) <= 0):
        if score > _MISSING_DISTANCE_SCORE_CAP:
            log.warning(
                f"distance_km null voi transport_mode known -> "
                f"cap score {score:.1f} -> {_MISSING_DISTANCE_SCORE_CAP}"
            )
            score = _MISSING_DISTANCE_SCORE_CAP

    _UNKNOWN_MODE_SCORE_CAP = 45.0
    if not transport_mode_known:
        if score > _UNKNOWN_MODE_SCORE_CAP:
            log.warning(
                f"transport_mode UNKNOWN -> CO2e tinh bang default truck factor, "
                f"khong chinh xac. Cap score {score:.1f} -> {_UNKNOWN_MODE_SCORE_CAP}"
            )
            score = _UNKNOWN_MODE_SCORE_CAP

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

def calculate(
    doc: ExtractedDocument,
    vessel_eff: Optional[VesselEfficiencyResult] = None,
    aircraft_result=None,   # Optional[AircraftEmissionResult] — avoid circular import
) -> ESGScore:
    """
    Entry point chinh. Nhan ExtractedDocument, tra ve ESGScore.

    Parameters:
        doc:             ExtractedDocument da duoc merge va resolve conflict.
        vessel_eff:      Ket qua lookup EU MRV (chi co gia tri khi mode = sea).
        aircraft_result: Ket qua lookup ICAO CEC (chi co gia tri khi mode = air).

    Transport methodology (theo thu tu uu tien):
        Sea + vessel_eff (high/medium): EU MRV vessel-specific WtW factor.
        Air + aircraft_result:          ICAO CEC aircraft-specific TTW x WtW 1.230.
        Default:                         EPA GHG Hub 2025 Table 8 + WtW correction.

    Packaging: EPA GHG Hub 2025 Table 9.
    Scoring: CO2e intensity (kg CO2e / metric ton cargo), GLEC Framework v3.2 aligned.
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
            doc, transport_factor, vessel_eff, aircraft_result
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