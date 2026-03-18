"""
core/rag.py

Query layer giua calculator va ChromaDB.
Nhiem vu duy nhat: nhan query, tra ve emission factors da duoc parse.

Key improvement: doc gia tri tu metadata.value (duoc luu boi ingest.py)
thay vi dung regex _extract_first_number() tren text chunk.
Metadata lookup chinh xac 100%, khong co false positive.

Fallback: neu ChromaDB miss hoac chua ingest, dung hardcoded EPA values.
"""

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
    Conversion sang metric ton-km duoc xu ly trong calculator.py.
    """
    mode: TransportMode
    co2_per_ton_mile: float
    source: str
    unit: str = "kg CO2 per short ton-mile"
    raw_chunk: str = ""


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
# Hardcoded fallback — EPA GHG Hub 2025, Table 8 + Table 9
# Dung khi ChromaDB chua duoc ingest hoac query miss.
# ---------------------------------------------------------------------------

_EPA_TRANSPORT_FALLBACK: dict[TransportMode, float] = {
    TransportMode.TRUCK:  0.186,
    TransportMode.RAIL:   0.021,
    TransportMode.SEA:    0.077,
    TransportMode.AIR:    1.086,
}

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
        log.warning(f"ChromaDB unavailable: {e}. Using fallback factors.")
        return None


def _get_collection(name: str) -> chromadb.Collection | None:
    client = _get_chroma_client()
    if client is None:
        return None
    try:
        col = client.get_collection(name)
        if col.count() == 0:
            log.warning(f"Collection '{name}' is empty. Run ingest.py first.")
            return None
        return col
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Transport query
# ---------------------------------------------------------------------------

def query_transport_factor(mode: TransportMode) -> TransportFactor:
    """
    Query ChromaDB EPA collection lay emission factor cho transport mode.

    Strategy: filter by metadata (type=transport, mode=<mode>) -> lay value tu metadata.
    Metadata.value duoc luu chinh xac boi ingest.py tu Excel cell value.
    Khong dung regex — eliminating _extract_first_number() bug.

    Fallback: hardcoded EPA value neu ChromaDB miss.
    """
    collection = _get_collection(settings.chroma_collection_epa)

    if collection is not None:
        try:
            result = collection.get(
                where={"$and": [{"type": "transport"}, {"mode": mode.value}]},
                limit=1,
                include=["metadatas", "documents"],
            )

            if result["metadatas"]:
                meta = result["metadatas"][0]
                val  = meta.get("value")
                if isinstance(val, (int, float)) and val > 0:
                    log.info(
                        f"RAG transport [{mode.value}]: {val} kg CO2/short ton-mile "
                        f"from {meta.get('source', 'EPA')} (metadata lookup)"
                    )
                    return TransportFactor(
                        mode=mode,
                        co2_per_ton_mile=float(val),
                        source=f"EPA_GHG_Hub_2025_Table8 | {meta.get('vehicle', '')}",
                        raw_chunk=result["documents"][0] if result["documents"] else "",
                    )
        except Exception as e:
            log.warning(f"ChromaDB transport query failed [{mode.value}]: {e}")

    # Fallback
    fallback_val = _EPA_TRANSPORT_FALLBACK.get(mode, _EPA_TRANSPORT_FALLBACK[TransportMode.TRUCK])
    log.debug(f"Fallback transport [{mode.value}]: {fallback_val}")
    return TransportFactor(
        mode=mode,
        co2_per_ton_mile=fallback_val,
        source="EPA_GHG_Hub_2025_hardcoded",
    )


# ---------------------------------------------------------------------------
# Packaging query
# ---------------------------------------------------------------------------

def query_packaging_factor(
    material: PackagingMaterial,
    disposal: DisposalMethod,
) -> PackagingFactor:
    """
    Query ChromaDB EPA collection lay emission factor cho packaging material + disposal.

    Strategy: filter by metadata (type=packaging, material=<material>, disposal=<disposal>)
    -> lay value tu metadata. Exact match, no ambiguity.

    Fallback: hardcoded EPA value neu ChromaDB miss.
    """
    effective_disposal = disposal if disposal != DisposalMethod.UNKNOWN else _DISPOSAL_FALLBACK

    collection = _get_collection(settings.chroma_collection_epa)

    if collection is not None:
        try:
            result = collection.get(
                where={
                    "$and": [
                        {"type":     "packaging"},
                        {"material": material.value},
                        {"disposal": effective_disposal.value},
                    ]
                },
                limit=1,
                include=["metadatas", "documents"],
            )

            if result["metadatas"]:
                meta = result["metadatas"][0]
                val  = meta.get("value")
                if isinstance(val, (int, float)) and val >= 0:
                    log.info(
                        f"RAG packaging [{material.value}/{effective_disposal.value}]: "
                        f"{val} metric tons CO2e/short ton "
                        f"from {meta.get('source', 'EPA')} (metadata lookup)"
                    )
                    return PackagingFactor(
                        material=material,
                        disposal=effective_disposal,
                        co2e_per_ton=float(val),
                        source=(
                            f"EPA_GHG_Hub_2025_Table9 | "
                            f"{meta.get('raw_material', material.value)}"
                        ),
                        raw_chunk=result["documents"][0] if result["documents"] else "",
                    )
        except Exception as e:
            log.warning(f"ChromaDB packaging query failed [{material.value}]: {e}")

    # Fallback
    fallback_val = _EPA_PACKAGING_FALLBACK.get(
        (material, effective_disposal),
        _EPA_PACKAGING_FALLBACK[(PackagingMaterial.MIXED_PLASTICS, DisposalMethod.LANDFILLED)],
    )
    log.debug(f"Fallback packaging [{material.value}/{effective_disposal.value}]: {fallback_val}")
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