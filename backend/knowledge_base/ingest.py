"""
knowledge_base/ingest.py

Chay mot lan de build ChromaDB tu GLEC PDF va EPA Excel.
Sau do khong can chay lai tru khi co version moi cua 2 file nay.

Usage:
    python -m backend.knowledge_base.ingest
"""

import re
import sys
import logging
from pathlib import Path

import chromadb
import openpyxl
from chromadb.api import ClientAPI
from pypdf import PdfReader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

settings = get_settings()

RAW_DIR = Path(__file__).parent / "raw"
GLEC_PDF = RAW_DIR / "GLEC_FRAMEWORK_v3_23_10_24_B.pdf"
EPA_XLSX = RAW_DIR / "ghg-emission-factors-hub-2025.xlsx"


# ---------------------------------------------------------------------------
# ChromaDB client
# ---------------------------------------------------------------------------

def get_chroma_client() -> ClientAPI:
    Path(settings.chroma_persist_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_persist_path)


# ---------------------------------------------------------------------------
# GLEC ingestion
# ---------------------------------------------------------------------------

# Nhung section trong GLEC lien quan den transport emission intensity.
# Chi ingest nhung section nay — bo qua phan governance, glossary, etc.
# de giam noise khi RAG query.
GLEC_RELEVANT_SECTIONS = [
    "road transport",
    "sea transport", "ocean", "maritime",
    "air transport", "aviation",
    "rail transport",
    "emission intensity",
    "default emission factor",
    "tce",                      # Transport Chain Element
    "gco2e", "co2e",
    "tonne-km", "ton-km",
]


def _is_relevant_glec_chunk(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in GLEC_RELEVANT_SECTIONS)


def _chunk_glec_pdf(path: Path, chunk_size: int = 800) -> list[dict]:
    """
    Doc GLEC PDF va chunk theo sliding window tren tung trang.

    chunk_size = 800 chars la diem can bang giua:
    - Du context cho LLM hieu emission factor
    - Khong qua lon khien embedding mat precision

    Moi chunk giu them page_number de debug khi RAG tra ve sai.
    """
    reader = PdfReader(str(path))
    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        # Sliding window voi 10% overlap de tranh cat doan giua con so va don vi
        overlap = int(chunk_size * 0.10)
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size].strip()
            if len(chunk) < 80:         # bo qua chunk qua ngan (header/footer)
                continue
            if not _is_relevant_glec_chunk(chunk):
                continue
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": "GLEC_Framework_v3.2",
                    "page": page_num,
                    "type": "transport_emission",
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
    log.info(f"GLEC: ingested {len(chunks)} chunks vao collection '{settings.chroma_collection_glec}'")


# ---------------------------------------------------------------------------
# EPA ingestion
# ---------------------------------------------------------------------------

# Chi lay Table 8 (transport) va Table 9 (packaging/waste).
# Cac table khac (combustion, electricity, refrigerants...) khong lien quan
# den bai toan logistics cua chung ta.
EPA_TARGET_TABLES = {
    "Table 8": "transport",
    "Table 9": "packaging",
}


def _extract_epa_tables(path: Path) -> list[dict]:
    """
    Doc EPA Excel va convert tung row cua Table 8 + Table 9
    thanh natural language chunk de ChromaDB embed.

    Li do dung natural language thay vi raw CSV:
    - LLM query bang text ("emission factor for corrugated containers recycled")
    - Natural language embedding match tot hon structured data
    """
    wb = openpyxl.load_workbook(str(path))
    ws = wb.active
    if ws is None:
        log.warning("Khong tim thay worksheet active trong file EPA XLSX.")
        return []
    rows = list(ws.iter_rows(values_only=True))

    chunks = []
    current_table = None
    headers = []

    for row in rows:
        # Detect table header
        for cell in row:
            if cell and str(cell).strip() in EPA_TARGET_TABLES:
                current_table = str(cell).strip()
                headers = []
                break

        if current_table is None:
            continue

        # Detect next table ngoai target -> stop
        for cell in row:
            if (cell and str(cell).strip().startswith("Table ")
                    and str(cell).strip() not in EPA_TARGET_TABLES
                    and str(cell).strip() != current_table):
                current_table = None
                break

        if current_table is None:
            continue

        # Lay header row (dong co nhieu text nhat)
        non_none = [c for c in row if c is not None]
        if len(non_none) >= 3 and any(
            isinstance(c, str) and len(c) > 4 for c in non_none
        ):
            potential_headers = [str(c).strip() if c else "" for c in row]
            if any(kw in " ".join(potential_headers).lower()
                   for kw in ["vehicle", "material", "factor", "co2", "unit"]):
                headers = potential_headers
                continue

        # Data row: co it nhat 1 so
        numeric_vals = [c for c in row if isinstance(c, (int, float))]
        if not numeric_vals or not headers:
            continue

        material_or_vehicle = next(
            (str(c).strip() for c in row if isinstance(c, str) and len(str(c).strip()) > 2),
            None,
        )
        if not material_or_vehicle:
            continue

        # Build natural language chunk
        table_type = EPA_TARGET_TABLES[current_table]

        if table_type == "transport":
            # Table 8: Vehicle Type | CO2 Factor | CH4 Factor | N2O Factor | Units
            chunk = _build_transport_chunk(material_or_vehicle, row, headers)
        else:
            # Table 9: Material | Recycled | Landfilled | Combusted | ...
            chunk = _build_packaging_chunk(material_or_vehicle, row, headers)

        if chunk:
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": "EPA_GHG_Hub_2025",
                    "table": current_table,
                    "type": table_type,
                    "material_or_vehicle": material_or_vehicle,
                },
            })

    log.info(f"EPA: extracted {len(chunks)} chunks from Table 8 + Table 9")
    return chunks


def _build_transport_chunk(vehicle: str, row: tuple, headers: list) -> str | None:
    """
    Convert mot row Table 8 thanh natural language.
    Vi du output:
    'EPA Table 8 transport emission factor for Medium- and Heavy-Duty Truck:
     CO2 factor is 0.186 kg CO2 per short ton-mile.
     Unit: short ton-mile. Source: EPA GHG Hub 2025.'
    """
    # Tim CO2 factor va unit
    co2_val = None
    unit = None
    for i, cell in enumerate(row):
        if isinstance(cell, float) and 0.001 < cell < 10:
            header = headers[i] if i < len(headers) else ""
            if "co2" in header.lower() and co2_val is None:
                co2_val = cell
        if isinstance(cell, str) and ("mile" in cell.lower() or "ton" in cell.lower()):
            unit = cell.strip()

    if co2_val is None:
        return None

    return (
        f"EPA Table 8 transport emission factor for {vehicle}: "
        f"CO2 factor is {co2_val} kg CO2 per {unit or 'short ton-mile'}. "
        f"Source: EPA Emission Factors for GHG Inventories 2025."
    )


def _build_packaging_chunk(material: str, row: tuple, headers: list) -> str | None:
    """
    Convert mot row Table 9 thanh natural language.
    Vi du output:
    'EPA Table 9 packaging emission factor for Corrugated Containers:
     Recycled disposal: 0.11 metric tons CO2e per short ton material.
     Landfilled disposal: 1.00 metric tons CO2e per short ton material.
     Source: EPA GHG Hub 2025.'

    NOTE: Recycled vs Landfilled chênh lệch rat lon (vi du carton: 0.11 vs 1.00).
    Chunk phai giu ca hai gia tri de RAG tra ve du thong tin cho calculator
    tinh dung theo disposal_method ma LLM extract duoc tu chung tu.
    """
    recycled_val = None
    landfilled_val = None

    for i, cell in enumerate(row):
        if not isinstance(cell, (int, float)):
            continue
        header = headers[i].lower() if i < len(headers) else ""
        if "recycl" in header and recycled_val is None:
            recycled_val = float(cell)
        elif "landfill" in header and landfilled_val is None:
            landfilled_val = float(cell)

    if recycled_val is None and landfilled_val is None:
        return None

    parts = [f"EPA Table 9 packaging emission factor for {material}:"]
    if recycled_val is not None:
        parts.append(f"Recycled disposal: {recycled_val} metric tons CO2e per short ton material.")
    if landfilled_val is not None:
        parts.append(f"Landfilled disposal: {landfilled_val} metric tons CO2e per short ton material.")
    parts.append("Source: EPA Emission Factors for GHG Inventories 2025.")

    return " ".join(parts)


def ingest_epa(client: ClientAPI) -> None:
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_epa,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0:
        log.info(f"EPA collection da co {existing} chunks, skip. Dung --force de re-ingest.")
        return

    chunks = _extract_epa_tables(EPA_XLSX)
    if not chunks:
        log.warning("Khong tim thay chunks EPA — kiem tra lai file XLSX.")
        return

    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[f"epa_{i}" for i in range(len(chunks))],
    )
    log.info(f"EPA: ingested {len(chunks)} chunks vao collection '{settings.chroma_collection_epa}'")


# ---------------------------------------------------------------------------
# Force re-ingest
# ---------------------------------------------------------------------------

def force_reingest(client: ClientAPI) -> None:
    """
    Xoa ca 2 collection va ingest lai tu dau.
    Dung khi update file GLEC hoac EPA len version moi.
    """
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

    # Summary
    for name in [settings.chroma_collection_glec, settings.chroma_collection_epa]:
        col = client.get_collection(name)
        log.info(f"Collection '{name}': {col.count()} chunks")

    log.info("Ingest complete.")