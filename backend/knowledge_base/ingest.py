"""
knowledge_base/ingest.py

Chay mot lan de build ChromaDB tu GLEC PDF va EPA Excel.
Sau do khong can chay lai tru khi co version moi cua 2 file nay.

Key improvement: luu gia tri emission factor vao metadata (value field)
thay vi chi embed natural language text. RAG query theo metadata.value
thay vi parse text bang regex -> accurate 100%.

Usage:
    python -m backend.knowledge_base.ingest
    python -m backend.knowledge_base.ingest --force   # re-ingest tu dau
"""

import sys
import logging
from pathlib import Path

import chromadb
import openpyxl
from chromadb.api import ClientAPI
from pypdf import PdfReader
import re

sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

settings = get_settings()

RAW_DIR   = Path(__file__).parent / "raw"
GLEC_PDF  = RAW_DIR / "GLEC_FRAMEWORK_v3_23_10_24_B.pdf"
EPA_XLSX  = RAW_DIR / "ghg-emission-factors-hub-2025.xlsx"


# ---------------------------------------------------------------------------
# ChromaDB client
# ---------------------------------------------------------------------------

def get_chroma_client() -> ClientAPI:
    Path(settings.chroma_persist_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_persist_path)


# ---------------------------------------------------------------------------
# EPA Table 8 — Transport factors
# ---------------------------------------------------------------------------
#
# Structure (row 418 in xlsx):
#   col 2: Vehicle Type
#   col 3: CO2 Factor (kg CO2 / unit)
#   col 6: Units (vehicle-mile | short ton-mile)
#
# Chi lay cac row co unit = "short ton-mile" (logistics freight, not passenger)
# Va map vehicle type -> TransportMode enum value de RAG query theo mode.

_TRANSPORT_MODE_MAP: dict[str, str] = {
    "medium- and heavy-duty truck":  "truck",
    "rail":                           "rail",
    "waterborne craft":               "sea",
    "aircraft":                       "air",
}


def _parse_table8(ws) -> list[dict]:
    """
    Parse Table 8 truc tiep theo row index da biet tu khao sat xlsx.
    Tra ve list records, moi record co:
    - text: natural language (cho embedding)
    - metadata: co value float va mode string (cho lookup chinh xac)
    """
    rows = list(ws.iter_rows(values_only=True))
    records = []

    # Table 8 bat dau o row 418 (0-indexed), header o row 420, data o row 421-428
    TABLE8_START = 418
    TABLE8_END   = 431  # row 429 = source footnote, 430 = notes, 431 = empty

    for row in rows[TABLE8_START:TABLE8_END]:
        vehicle = row[2]
        co2_val = row[3]
        unit    = row[6]

        if not isinstance(vehicle, str) or not isinstance(co2_val, (int, float)):
            continue
        if not isinstance(unit, str) or "short ton-mile" not in unit.lower():
            continue

        vehicle_clean = vehicle.strip().rstrip("ABC").strip().lower()
        mode = next(
            (m for k, m in _TRANSPORT_MODE_MAP.items() if k in vehicle_clean),
            None
        )
        if mode is None:
            continue

        co2_val = float(co2_val)

        text = (
            f"EPA Table 8 transport emission factor for {vehicle.strip()}: "
            f"CO2 factor is {co2_val} kg CO2 per short ton-mile. "
            f"Transport mode: {mode}. "
            f"Source: EPA Emission Factors for GHG Inventories 2025."
        )

        records.append({
            "text": text,
            "metadata": {
                "source":  "EPA_GHG_Hub_2025_Table8",
                "type":    "transport",
                "mode":    mode,
                "value":   co2_val,
                "unit":    "kg_co2_per_short_ton_mile",
                "vehicle": vehicle.strip(),
            },
        })
        log.info(f"Table 8: {mode} ({vehicle.strip()}) = {co2_val} kg CO2/short ton-mile")

    return records


# ---------------------------------------------------------------------------
# EPA Table 9 — Packaging factors
# ---------------------------------------------------------------------------
#
# Structure (row 431 in xlsx):
#   col 2: Material
#   col 3: Recycled (metric ton CO2e / short ton material)
#   col 4: Landfilled
#   col 5: Combusted
#   col 6: Composted
#
# Moi material tao 2 records: recycled va landfilled.
# Moi record co metadata.value de RAG lookup chinh xac.

_PACKAGING_MATERIAL_MAP: dict[str, str] = {
    "aluminum cans":         "aluminum",
    "steel cans":            "steel",
    "glass":                 "glass",
    "hdpe":                  "hdpe",
    "pet":                   "pet",
    "corrugated containers": "carton",
    "mixed metals":          "mixed_metals",
    "mixed plastics":        "mixed_plastics",
    "ldpe":                  "mixed_plastics",
    "lldpe":                 "mixed_plastics",
    "pp":                    "mixed_plastics",
}


def _parse_table9(ws) -> list[dict]:
    """
    Parse Table 9 truc tiep.
    Moi material + disposal_method -> 1 record voi metadata.value.
    """
    rows = list(ws.iter_rows(values_only=True))
    records = []

    # Table 9 bat dau o row 431, data o row 435 tro di
    TABLE9_START = 435
    TABLE9_END   = 490  # du de cover het material rows

    for row in rows[TABLE9_START:TABLE9_END]:
        material = row[2]
        recycled   = row[3]
        landfilled = row[4]

        if not isinstance(material, str) or len(material.strip()) < 2:
            continue

        material_clean = material.strip().lower()
        # Exact match first, then word-boundary regex
        # Prevents "carpet" -> "pet" and "copper wire" -> wrong mapping
        material_key = _PACKAGING_MATERIAL_MAP.get(material_clean)
        if material_key is None:
            import re as _re
            material_key = next(
                (v for k, v in _PACKAGING_MATERIAL_MAP.items()
                 if _re.search(r"\b" + _re.escape(k) + r"\b", material_clean)),
                None
            )
        if material_key is None:
            continue

        for disposal, val in [("recycled", recycled), ("landfilled", landfilled)]:
            if not isinstance(val, (int, float)):
                continue
            val = float(val)

            text = (
                f"EPA Table 9 packaging emission factor for {material.strip()}: "
                f"{disposal} disposal is {val} metric tons CO2e per short ton material. "
                f"Material type: {material_key}. "
                f"Source: EPA Emission Factors for GHG Inventories 2025."
            )

            records.append({
                "text": text,
                "metadata": {
                    "source":   "EPA_GHG_Hub_2025_Table9",
                    "type":     "packaging",
                    "material": material_key,
                    "disposal": disposal,
                    "value":    val,
                    "unit":     "metric_tons_co2e_per_short_ton_material",
                    "raw_material": material.strip(),
                },
            })

        log.info(f"Table 9: {material.strip()} -> {material_key}")

    return records


def ingest_epa(client: ClientAPI) -> None:
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_epa,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0:
        log.info(f"EPA collection da co {existing} chunks, skip. Dung --force de re-ingest.")
        return

    wb = openpyxl.load_workbook(str(EPA_XLSX))
    ws = wb.active

    transport_records = _parse_table8(ws)
    packaging_records = _parse_table9(ws)
    all_records = transport_records + packaging_records

    if not all_records:
        log.warning("Khong tim thay records EPA — kiem tra lai file XLSX.")
        return

    collection.add(
        documents=[r["text"] for r in all_records],
        metadatas=[r["metadata"] for r in all_records],
        ids=[f"epa_{i}" for i in range(len(all_records))],
    )
    log.info(
        f"EPA: ingested {len(transport_records)} transport + "
        f"{len(packaging_records)} packaging records"
    )


# ---------------------------------------------------------------------------
# GLEC PDF — contextual chunks (cho explanation generation)
# ---------------------------------------------------------------------------

GLEC_RELEVANT_SECTIONS = [
    "road transport", "sea transport", "ocean", "maritime",
    "air transport", "aviation", "rail transport",
    "emission intensity", "default emission factor",
    "tce", "gco2e", "co2e", "tonne-km", "ton-km",
    "well-to-wheel", "wtw", "tank-to-wheel", "ttw",
    "glec framework", "iso 14083",
]


def _chunk_glec_pdf(path: Path, chunk_size: int = 800) -> list[dict]:
    reader = PdfReader(str(path))
    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        text_lower = text.lower()
        if not any(kw in text_lower for kw in GLEC_RELEVANT_SECTIONS):
            continue

        overlap = int(chunk_size * 0.10)
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunk = text[i: i + chunk_size].strip()
            if len(chunk) < 80:
                continue
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": "GLEC_Framework_v3.2",
                    "page":   page_num,
                    "type":   "transport_emission",
                },
            })

    log.info(f"GLEC: extracted {len(chunks)} relevant chunks from {len(reader.pages)} pages")
    return chunks


def ingest_glec(client: ClientAPI) -> None:
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_glec,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0:
        log.info(f"GLEC collection da co {existing} chunks, skip. Dung --force de re-ingest.")
        return

    chunks = _chunk_glec_pdf(GLEC_PDF)
    if not chunks:
        log.warning("Khong tim thay chunks GLEC — kiem tra lai file PDF.")
        return

    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[f"glec_{i}" for i in range(len(chunks))],
    )
    log.info(f"GLEC: ingested {len(chunks)} chunks vao '{settings.chroma_collection_glec}'")


# ---------------------------------------------------------------------------
# Force re-ingest
# ---------------------------------------------------------------------------

def force_reingest(client: ClientAPI) -> None:
    for name in [settings.chroma_collection_glec, settings.chroma_collection_epa]:
        try:
            client.delete_collection(name)
            log.info(f"Deleted collection: {name}")
        except Exception:
            pass
    ingest_glec(client)
    ingest_epa(client)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    force = "--force" in sys.argv
    client = get_chroma_client()

    if force:
        log.info("Force re-ingest mode")
        force_reingest(client)
    else:
        ingest_glec(client)
        ingest_epa(client)

    for name in [settings.chroma_collection_glec, settings.chroma_collection_epa]:
        try:
            col = client.get_collection(name)
            log.info(f"Collection '{name}': {col.count()} records")
        except Exception:
            pass

    log.info("Ingest complete.")