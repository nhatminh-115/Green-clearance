"""
core/vessel_lookup.py

Logic for EU MRV Vessel Efficiency Scoring.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import pandas as pd
from rapidfuzz import fuzz, process

from backend.config import get_settings
from backend.models.schemas import VesselEfficiencyResult

log = logging.getLogger(__name__)
settings = get_settings()

_df: Optional[pd.DataFrame] = None
_ship_type_stats: Dict[str, pd.Series] = {}

def load_mrv_dataset() -> None:
    """Load EU MRV dataset vao memory, precalculate percentiles per ship type.
    Su dung pickle de cache dataframe parse tu excel giup tang toc do load (<0.1s thay vi 28s).
    """
    global _df, _ship_type_stats
    if _df is not None:
        return

    raw_path = Path("backend/knowledge_base/raw/2024-v195-20032026-EU MRV Publication of information.xlsx")
    cache_path = Path("backend/knowledge_base/processed/mrv_dataset.pkl")
    
    try:
        if cache_path.exists():
            log.info(f"Loading MRV dataset from cache {cache_path}...")
            df_clean = pd.read_pickle(cache_path)
            emissions_col = "emissions_intensity"
        else:
            if not raw_path.exists():
                log.warning(f"MRV dataset not found at {raw_path}. Vessel lookup will be gracefully disabled.")
                return
                
            log.info(f"Loading raw MRV dataset from {raw_path} (this may take ~20-30s)...")
            df_raw = pd.read_excel(raw_path, sheet_name="2024 Full ERs", skiprows=2)
            
            # Combine emission metrics (mass > freight > dwt > volume) de co nhieu data nhat
            col_candidates = [
                "CO₂ emissions per transport work (mass) [g CO₂ / m tonnes · n miles]",
                "CO₂ emissions per transport work (freight) [g CO₂ / m tonnes · n miles]",
                "CO₂ emissions per transport work (dwt) [g CO₂ / dwt carried · n miles]",
                "CO₂ emissions per transport work (volume) [g CO₂ / m³ · n miles]"
            ]
            
            # Ghep cac cot de lay duoc intensity
            df_raw['emissions_intensity'] = pd.NA
            for c in col_candidates:
                if c in df_raw.columns:
                    df_raw['emissions_intensity'] = df_raw['emissions_intensity'].combine_first(df_raw[c])
            
            # Coerce to numeric, turning 'Division by zero!' into NaN
            df_raw['emissions_intensity'] = pd.to_numeric(df_raw['emissions_intensity'], errors='coerce')
            
            # Filter non-null
            df_clean = df_raw.dropna(subset=["Name", "emissions_intensity"]).copy()
            df_clean["name_lower"] = df_clean["Name"].astype(str).str.lower().str.strip()
            emissions_col = "emissions_intensity"
            
            # Luu cache cho lan sau
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df_clean.to_pickle(cache_path)
            log.info(f"Saved parsed MRV dataset to cache at {cache_path}.")

        # Pre-calculate percentiles distribution for each ship type
        for ship_type, group in df_clean.groupby("Ship type"):
            # Compute percentiles cho grading A-E
            _ship_type_stats[ship_type] = group[emissions_col].astype(float).copy()
            
        _df = df_clean
        log.info(f"Loaded {_df.shape[0]} vessels from MRV dataset successfully.")
    except Exception as e:
        log.warning(f"Failed to load MRV dataset: {e}. Vessel lookup disabled.")


def _infer_ship_type_from_cargo(cargo_type: str) -> str:
    """Map common cargo types to EU MRV Ship types."""
    c = str(cargo_type).lower()
    if "container" in c:
        return "Container ship"
    elif "bulk" in c or "coal" in c or "grain" in c:
        return "Bulk carrier"
    elif "oil" in c or "petroleum" in c:
        return "Oil tanker"
    elif "gas" in c or "lng" in c or "lpg" in c:
        return "Gas carrier"
    elif "vehicle" in c or "car" in c or "ro-ro" in c:
        return "Ro-ro ship"
    return "General cargo ship"


def _parse_eedi(tech_eff: str) -> Optional[float]:
    """Parse string 'EEDI (4.12 gCO₂/t·nm)' to get 4.12."""
    if not isinstance(tech_eff, str):
        return None
    match = re.search(r"([\d\.]+)", str(tech_eff))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _get_grade_from_percentile(percentile: float) -> str:
    """Return grade A-E based on percentile (0 is best, 100 is worst)."""
    if pd.isna(percentile):
        return "C"  # Default fallback if NaN
    if percentile <= 20:
        return "A"
    elif percentile <= 40:
        return "B"
    elif percentile <= 60:
        return "C"
    elif percentile <= 80:
        return "D"
    return "E"


def _get_grade_from_eedi(eedi_value: float, ship_type: str) -> str:
    """
    Fallback: Estimate grade based on EEDI. 
    MEPC.203(62) has complex baselines per ship type and deadweight. 
    We approximate based on general heuristics relative to EU MRV averages.
    """
    # Simply map absolute EEDI values to grades as a fallback heuristic
    if eedi_value < 5.0:
        return "A"
    elif eedi_value < 8.0:
        return "B"
    elif eedi_value < 12.0:
        return "C"
    elif eedi_value < 18.0:
        return "D"
    return "E"


def lookup_vessel_efficiency(
    vessel_name: Optional[str] = None, 
    carrier_name: Optional[str] = None, 
    voyage_number: Optional[str] = None, 
    cargo_type: Optional[str] = None
) -> VesselEfficiencyResult:
    """
    Lookup vessel in MRV dataset using multi-signal disambiguation.
    Returns a VesselEfficiencyResult object.
    """
    # Attempt to lazily load dataset if needed
    if _df is None:
        load_mrv_dataset()
        
    if _df is None or _df.empty or not vessel_name:
        return VesselEfficiencyResult(
            vessel_name_matched=str(vessel_name or "Unknown"),
            confidence_level="none",
            grade_source="no_data"
        )

    v_name_clean = vessel_name.lower().strip()
    c_name_clean = carrier_name.lower().strip() if carrier_name else ""
    expected_ship_type = _infer_ship_type_from_cargo(cargo_type) if cargo_type else None

    # Get top 30 fuzzy matches by name first to reduce search space
    names_list = _df["name_lower"].to_list()
    top_matches = process.extract(v_name_clean, names_list, limit=30, scorer=fuzz.token_sort_ratio)
    
    # -----------------------------------------------------------------------
    # STRICT NAME GATE: vessel name must match >= 90% before any other signal
    # is considered. This prevents partial token matches (e.g. 'MSC FORTUNATE'
    # matching 'SAGA FORTUNE') from producing wrong-vessel grades.
    # -----------------------------------------------------------------------
    _MIN_NAME_RATIO = 90.0

    # Quick check: is the best possible name match good enough?
    best_ratio_overall = top_matches[0][1] if top_matches else 0.0
    if best_ratio_overall < _MIN_NAME_RATIO:
        # No vessel in database is close enough — fall back to industry average
        return VesselEfficiencyResult(
            vessel_name_matched=str(vessel_name),
            confidence_level="low",
            efficiency_grade="C",
            grade_source="industry_average",
            percentile_in_ship_type=50.0
        )

    best_idx = None
    best_score = -1.0
    second_best_score = -1.0
    best_row: Optional[pd.Series] = None
    
    # Iterate through candidates and compute composite score
    # Only consider candidates that pass the strict name gate
    for match_name, match_ratio, original_idx in top_matches:
        if match_ratio < _MIN_NAME_RATIO:
            continue  # Skip weak name matches — they can't win on carrier/type alone
        row = _df.iloc[original_idx]
        
        # 1. Name match (40%)
        name_score = match_ratio * 0.40
        
        # 2. Carrier/Operator match (35%)
        # EU MRV "Company name" column actually stores the VERIFIER (DNV, Bureau Veritas, etc.)
        # NOT the shipping operator. We use it only when it looks like a real operator name.
        _VERIFIER_KEYWORDS = {
            'dnv', 'bureau veritas', 'lloyd', 'rina', 'abs', 'bv', 'ccs', 'nk',
            'class nk', 'tuv', 'germanischer lloyd', 'korean register', 'rmrs',
        }
        company_col = "Company name" if "Company name" in row else "Verifier Name"
        db_carrier = str(row.get(company_col, "")).lower().strip()
        
        carrier_score = 0.0
        # Check if the db company value is a blank, a placeholder, or a classification society (verifier)
        is_verifier_or_blank = (
            not db_carrier
            or db_carrier in ('-', '--', 'nan', 'none', '')
            or any(kw in db_carrier for kw in _VERIFIER_KEYWORDS)
        )
        if c_name_clean:
            if is_verifier_or_blank:
                # Can't meaningfully compare carrier vs. verifier — treat as neutral
                carrier_score = 20.0
            else:
                # Use partial_ratio so abbreviations like 'MSC' can match within
                # 'mediterranean shipping company s.a.' without full-string mismatch.
                partial = fuzz.partial_ratio(c_name_clean, db_carrier)
                # Acronym detection: check if carrier_name matches initials of company
                words = [w for w in db_carrier.split() if len(w) > 1 and w not in ('and', 'of', 'the', 's.a.')]
                acronym = ''.join(w[0] for w in words)
                acronym_match = 100 if c_name_clean == acronym else 0
                best_carrier_ratio = max(partial, acronym_match)
                carrier_score = best_carrier_ratio * 0.35
        else:
            # If carrier not provided in input, assume neutral 20/35 points to not excessively penalize
            carrier_score = 20.0
            
        # 3. Ship type consistency (15%)
        db_type = str(row.get("Ship type", ""))
        ship_type_score = 0.0
        if expected_ship_type:
            if expected_ship_type == db_type:
                ship_type_score = 15.0
            else:
                # Partial if vaguely similar (e.g. both cargo)
                if "cargo" in expected_ship_type.lower() and "cargo" in db_type.lower():
                    ship_type_score = 7.5
        else:
            # No cargo provided, neutral 10/15 points
            ship_type_score = 10.0
            
        # 4. Home port region match (10%)
        # We don't have home port readily available from standard logistics extract docs
        # Assign neutral 5/10 points
        home_port_score = 5.0
        
        composite_score = name_score + carrier_score + ship_type_score + home_port_score
        
        if composite_score > best_score:
            second_best_score = best_score
            best_score = composite_score
            best_idx = original_idx
            best_row = row
        elif composite_score > second_best_score:
            second_best_score = composite_score

    # Determine confidence level based on Decision rule
    margin = best_score - second_best_score
    
    if best_score >= 80.0 and margin >= 15.0:
        confidence = "high"
    elif best_score >= 60.0:
        confidence = "medium"
    else:
        confidence = "low"

    emissions_col = "emissions_intensity"

    # Populate result
    result = VesselEfficiencyResult(
        vessel_name_matched=str(best_row["Name"]) if best_row is not None else vessel_name,
        confidence_level=confidence,
        grade_source="no_data"
    )

    if confidence in ("high", "medium") and best_row is not None:
        imo = best_row.get("IMO Number", None)
        result.imo_number = int(imo) if pd.notna(imo) else None
        
        intensity = best_row.get(emissions_col)
        ship_type = best_row.get("Ship type", "General cargo ship")
        
        if pd.notna(intensity):
            result.emission_intensity_g_per_tonne_nm = float(intensity)
            
            # Look up percentile within the same ship type
            if ship_type in _ship_type_stats:
                series = _ship_type_stats[ship_type]
                # pct_of_data_smaller = percentile ranking
                pctrank = (series < intensity).mean() * 100.0
                result.percentile_in_ship_type = float(pctrank)
                result.efficiency_grade = _get_grade_from_percentile(pctrank)
                result.grade_source = "eu_mrv_actual"
                
    elif confidence == "low":
        # Khong dung best_row (vi confidence low la match sai tau)
        cargo_type_str = str(cargo_type or "")
        ship_type_inferred = _infer_ship_type_from_cargo(cargo_type_str)
        
        # Fallback to industry average
        result.vessel_name_matched = str(vessel_name) if vessel_name else "Unknown"
        result.imo_number = None
        result.efficiency_grade = "C"
        result.grade_source = "industry_average"
        result.percentile_in_ship_type = 50.0
            
    return result
