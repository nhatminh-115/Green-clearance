"""
core/rag.py

Query layer giua calculator va ChromaDB.
Nhiem vu duy nhat: nhan query text, tra ve emission factors da duoc parse.

Khong chua business logic tinh CO2e — do la viec cua calculator.py.
"""

import re
import logging
from dataclasses import dataclass, field
from functools import lru_cache

import chromadb

from backend.config import get_settings
from backend.models.schemas import TransportMode, PackagingMaterial, DisposalMethod

log = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class TransportFactor:
    """
    Emission factor cho mot transport mode.
    Don vi: kg CO2 per short ton-mile (EPA Table 8).

    NOTE: EPA dung short ton-mile, GLEC dung gCO2e/tonne-km.
    Conversion duoc xu ly trong calculator.py, khong phai o day.
    Rag chi tra ve raw value va don vi de calculator biet ma convert.
    """
    mode: TransportMode
    co2_per_ton_mile: float
    source: str
    unit: str = "kg CO2 per short ton-mile"
    raw_chunk: str = ""             # giu lai de trace khi debug


@dataclass
class PackagingFactor:
    """
    Emission factor cho mot loai vat lieu dong goi.
    Don vi: metric tons CO2e per short ton material (EPA Table 9).
    """
    material: PackagingMaterial
    disposal: DisposalMethod
    co2e_per_ton: float
    source: str
    unit: str = "metric tons CO2e per short ton material"
    raw_chunk: str = ""


@dataclass
class RAGResult:
    transport_factors: list[TransportFactor] = field(default_factory=list)
    packaging_factors: list[PackagingFactor] = field(default_factory=list)
    missing_transport: list[TransportMode] = field(default_factory=list)
    missing_packaging: list[tuple[PackagingMaterial, DisposalMethod]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hardcoded fallback factors
#
# EPA Table 8 va Table 9 values duoc hardcode lam fallback khi ChromaDB
# khong tim thay ket qua phu hop.
#
# Ly do can fallback:
# - Ingest chua chay hoac ChromaDB bi corrupt
# - Query text khong match chunk du dung material dung
# - Demo mode khong co ChromaDB
#
# Values lay truc tiep tu EPA GHG Hub 2025, Table 8 + Table 9.
# ---------------------------------------------------------------------------

_EPA_TRANSPORT_FALLBACK: dict[TransportMode, float] = {
    TransportMode.TRUCK:  0.186,   # Medium- and Heavy-Duty Truck, short ton-mile
    TransportMode.RAIL:   0.021,
    TransportMode.SEA:    0.077,   # Waterborne Craft
    TransportMode.AIR:    1.086,   # Aircraft
}

# (material, disposal) -> co2e_per_short_ton
_EPA_PACKAGING_FALLBACK: dict[tuple[PackagingMaterial, DisposalMethod], float] = {
    (PackagingMaterial.CARTON,         DisposalMethod.RECYCLED):   0.11,
    (PackagingMaterial.CARTON,         DisposalMethod.LANDFILLED):  1.00,
    (PackagingMaterial.HDPE,           DisposalMethod.RECYCLED):   0.21,
    (PackagingMaterial.HDPE,           DisposalMethod.LANDFILLED):  0.02,
    (PackagingMaterial.PET,            DisposalMethod.RECYCLED):   0.23,
    (PackagingMaterial.PET,            DisposalMethod.LANDFILLED):  0.02,
    (PackagingMaterial.MIXED_PLASTICS, DisposalMethod.RECYCLED):   0.22,
    (PackagingMaterial.MIXED_PLASTICS, DisposalMethod.LANDFILLED):  0.02,
    (PackagingMaterial.MIXED_METALS,   DisposalMethod.RECYCLED):   0.23,
    (PackagingMaterial.MIXED_METALS,   DisposalMethod.LANDFILLED):  0.02,
    (PackagingMaterial.STEEL,          DisposalMethod.RECYCLED):   0.32,
    (PackagingMaterial.STEEL,          DisposalMethod.LANDFILLED):  0.02,
    (PackagingMaterial.ALUMINUM,       DisposalMethod.RECYCLED):   0.06,
    (PackagingMaterial.ALUMINUM,       DisposalMethod.LANDFILLED):  0.02,
    (PackagingMaterial.GLASS,          DisposalMethod.RECYCLED):   0.05,
    (PackagingMaterial.GLASS,          DisposalMethod.LANDFILLED):  0.02,
}

# Unknown disposal -> conservative: dung landfilled
_DISPOSAL_FALLBACK = DisposalMethod.LANDFILLED


# ---------------------------------------------------------------------------
# ChromaDB client — singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_chroma_client():
    try:
        client = chromadb.PersistentClient(path=settings.chroma_persist_path)
        return client
    except Exception as e:
        log.warning(f"Khong ket duoc ChromaDB: {e}. Se dung fallback factors.")
        return None


def _get_collection(name: str) -> chromadb.Collection | None:
    client = _get_chroma_client()
    if client is None:
        return None
    try:
        col = client.get_collection(name)
        if col.count() == 0:
            log.warning(f"Collection '{name}' rong. Chay ingest.py truoc.")
            return None
        return col
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Number extraction helper
# ---------------------------------------------------------------------------

def _extract_first_number(text: str) -> float | None:
    """
    Lay so dau tien xuat hien trong text chunk.
    Dung regex don gian vi format EPA chunk la deterministic
    (do chinh ingest.py tao ra, khong phai free-form text).
    """
    matches = re.findall(r"\b\d+\.\d+|\b\d+\b", text)
    for m in matches:
        try:
            val = float(m)
            # Loc nhung so hop le cho emission factor
            # (qua nho hoac qua lon thi sai)
            if 0.001 < val < 100:
                return val
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Transport query
# ---------------------------------------------------------------------------

_TRANSPORT_QUERY_TEMPLATES: dict[TransportMode, str] = {
    TransportMode.SEA:   "emission factor waterborne craft sea ocean shipping short ton-mile CO2",
    TransportMode.AIR:   "emission factor aircraft air transport aviation short ton-mile CO2",
    TransportMode.TRUCK: "emission factor medium heavy duty truck road transport short ton-mile CO2",
    TransportMode.RAIL:  "emission factor rail transport short ton-mile CO2",
}


def query_transport_factor(mode: TransportMode) -> TransportFactor:
    """
    Query ChromaDB lay emission factor cho transport mode.
    Fallback ve EPA hardcoded value neu ChromaDB khong tra ket qua.
    """
    collection = _get_collection(settings.chroma_collection_epa)

    if collection is not None and mode in _TRANSPORT_QUERY_TEMPLATES:
        query = _TRANSPORT_QUERY_TEMPLATES[mode]
        try:
            result = collection.query(
                query_texts=[query],
                n_results=settings.rag_top_k,
                where={"type": "transport"},
            )
            chunks: list[str] = result["documents"][0] if result["documents"] else []

            for chunk in chunks:
                val = _extract_first_number(chunk)
                if val is not None:
                    log.debug(f"RAG transport [{mode}]: {val} from chunk")
                    return TransportFactor(
                        mode=mode,
                        co2_per_ton_mile=val,
                        source="EPA_GHG_Hub_2025_ChromaDB",
                        raw_chunk=chunk,
                    )
        except Exception as e:
            log.warning(f"ChromaDB query loi [{mode}]: {e}")

    # Fallback
    fallback_val = _EPA_TRANSPORT_FALLBACK.get(mode)
    if fallback_val is None:
        log.error(f"Khong co fallback factor cho transport mode: {mode}")
        fallback_val = _EPA_TRANSPORT_FALLBACK[TransportMode.TRUCK]  # worst case

    log.debug(f"Fallback transport [{mode}]: {fallback_val}")
    return TransportFactor(
        mode=mode,
        co2_per_ton_mile=fallback_val,
        source="EPA_GHG_Hub_2025_hardcoded",
    )


# ---------------------------------------------------------------------------
# Packaging query
# ---------------------------------------------------------------------------

_PACKAGING_QUERY_TEMPLATES: dict[PackagingMaterial, str] = {
    PackagingMaterial.CARTON:         "corrugated containers packaging emission factor recycled landfilled CO2e",
    PackagingMaterial.HDPE:           "HDPE plastic packaging emission factor recycled landfilled CO2e",
    PackagingMaterial.PET:            "PET plastic packaging emission factor recycled landfilled CO2e",
    PackagingMaterial.MIXED_PLASTICS: "mixed plastics packaging emission factor recycled landfilled CO2e",
    PackagingMaterial.MIXED_METALS:   "mixed metals packaging emission factor recycled landfilled CO2e",
    PackagingMaterial.STEEL:          "steel cans packaging emission factor recycled landfilled CO2e",
    PackagingMaterial.ALUMINUM:       "aluminum cans packaging emission factor recycled landfilled CO2e",
    PackagingMaterial.GLASS:          "glass packaging emission factor recycled landfilled CO2e",
}

_DISPOSAL_KEYWORDS = {
    DisposalMethod.RECYCLED:   "recycled",
    DisposalMethod.LANDFILLED: "landfilled",
}


def query_packaging_factor(
    material: PackagingMaterial,
    disposal: DisposalMethod,
) -> PackagingFactor:
    effective_disposal = disposal
    if disposal == DisposalMethod.UNKNOWN:
        effective_disposal = _DISPOSAL_FALLBACK

    # Dung thang fallback cho packaging — EPA Table 9 la fixed data
    # RAG unreliable vi _extract_first_number hay pick wrong number tu chunk
    fallback_val = _EPA_PACKAGING_FALLBACK.get((material, effective_disposal))
    if fallback_val is None:
        fallback_val = _EPA_PACKAGING_FALLBACK[
            (PackagingMaterial.MIXED_PLASTICS, DisposalMethod.LANDFILLED)
        ]
        log.warning(f"Khong co fallback cho ({material}, {effective_disposal})")

    return PackagingFactor(
        material=material,
        disposal=effective_disposal,
        co2e_per_ton=fallback_val,
        source="EPA_GHG_Hub_2025_hardcoded",
    )


# ---------------------------------------------------------------------------
# Batch query — entry point chinh cho calculator.py
# ---------------------------------------------------------------------------

def query_all_factors(
    transport_modes: list[TransportMode],
    packaging_requests: list[tuple[PackagingMaterial, DisposalMethod]],
) -> RAGResult:
    """
    Query mot lan cho tat ca modes va materials can thiet.
    Calculator goi ham nay thay vi goi tung ham rieng le.
    """
    result = RAGResult()

    for mode in transport_modes:
        if mode == TransportMode.UNKNOWN:
            result.missing_transport.append(mode)
            continue
        factor = query_transport_factor(mode)
        result.transport_factors.append(factor)

    for material, disposal in packaging_requests:
        if material == PackagingMaterial.UNKNOWN:
            result.missing_packaging.append((material, disposal))
            continue
        factor = query_packaging_factor(material, disposal)
        result.packaging_factors.append(factor)

    return result