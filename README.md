# Green Clearance

Logistics emissions scoring tool. Upload trade documents (Commercial Invoice, Packing List, Bill of Lading, TDS, PPWR), get CO2e emissions calculated and an ESG score with GREEN / YELLOW / RED lane classification.

Built for the CAIEC competition — Transportation & Logistics category.

---

## What it does

1. Reads uploaded documents via vision AI (Groq llama-4-scout)
2. Classifies each document type automatically (hybrid keyword + LLM)
3. Extracts structured data — transport mode, ports, cargo weight, packaging materials
4. Merges data from multiple documents with conflict detection and resolution
5. Estimates distance via searoute-py (sea) or Haversine + DEFRA factor (air/truck)
6. Calculates CO2e emissions aligned with GLEC Framework v3.2 / ISO 14083
7. Outputs an ESG score (0–100) with lane classification and natural-language explanation
8. Logs results to Supabase for history tracking

---

## Stack

- **Backend:** FastAPI + LangGraph
- **AI/LLM:** Groq (llama-4-scout-17b for vision, llama-3.3-70b for classification/explanation)
- **Distance (sea):** searoute-py — actual sea routing graph, accounts for straits
- **Distance (air):** Haversine x 1.09 (DEFRA GHG Conversion Factors) / multi-leg from AWB routing stops
- **Distance (truck/rail):** Haversine x road detour factor
- **Fuzzy port matching:** rapidfuzz
- **Database:** Supabase (audit log + history)
- **Frontend:** Single HTML file, no framework

---

## Methodology

### Emissions calculation

Transport CO2e follows GLEC Framework v3.2 / ISO 14083:

```
CO2e = cargo_weight_tons x distance_km x EPA_factor x WtW_ratio
```

Emission factors from **EPA GHG Hub 2025, Table 8** (TTW, kg CO2/short ton-mile), converted to metric ton-km. WtW correction applied per GLEC v3.2 Module 1:

| Mode  | EPA factor (TTW) | WtW ratio | Source |
|-------|-----------------|-----------|--------|
| Sea   | 0.077           | 1.215     | IMO MEPC 81 (VLSFO) |
| Air   | 1.086           | 1.230     | GREET 2023 (Jet A proxy) |
| Truck | 0.186           | 1.202     | GREET 2023 (Diesel) |
| Rail  | 0.021           | 1.202     | GREET 2023 (Diesel) |

Packaging CO2e from **EPA GHG Hub 2025, Table 9** (metric ton CO2e/short ton material by disposal method).

### Scoring

Score is based on **CO2e intensity** (kg CO2e per metric ton of cargo), not absolute emissions. A 50t sea shipment emitting 50 tCO2e scores higher than a 2t air shipment emitting 20 tCO2e because intensity per ton is what matters.

```
intensity = total_co2e_kg / cargo_weight_tons
score     = (10,000 - intensity) / (10,000 - 200) x 100
```

The thresholds (200–10,000 kg/ton) are calibrated to the actual range of logistics operations — best-case sea short haul to worst-case air long haul. Proprietary scoring scale, not an industry standard.

| Lane   | Score | Typical intensity |
|--------|-------|-------------------|
| GREEN  | >= 70 | < ~3,100 kg/ton   |
| YELLOW | 40–69 | ~3,100–6,200 kg/ton |
| RED    | < 40  | > ~6,200 kg/ton   |

### Distance estimation

When documents do not include distance:

- **Sea:** searoute-py (actual sea routing, accounts for Malacca, Suez, Panama)
- **Air direct:** Haversine x 1.09 — DEFRA GHG Conversion Factors methodology
- **Air multi-leg:** Sum of per-leg Haversine x 1.02 (ATC deviation per leg), routing extracted from AWB
- **Truck:** Haversine x 1.18–1.35 (road detour factor by distance)
- **Rail:** Haversine x 1.20

### Document merge priority

```
BL(1) > CI(2) > PL(3) > TDS(4) > PPWR(5)
```

Exception: `cargo_weight_tons` uses **PL(1) > CI(2) > BL(3)** — BL reports gross weight (cargo + packaging), not net weight needed for emissions.

Packaging weight: PL only. Disposal method: PPWR > TDS > PL.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys

# Build ChromaDB from EPA + GLEC source files (run once)
python -m backend.knowledge_base.ingest

# Re-ingest when EPA or GLEC files are updated to a new version
python -m backend.knowledge_base.ingest --force

uvicorn backend.main:app --reload
```

Open `frontend/app.html` in a browser (or serve via any static file server).

### Environment variables

```
GROQ_API_KEY=
SUPABASE_URL=
SUPABASE_KEY=
CHROMA_PERSIST_PATH=./backend/knowledge_base/chroma_db
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

---

## Project structure

```
green_clearance/
├── backend/
│   ├── api/routes/
│   │   ├── upload.py          # POST /api/v1/upload, /upload/multi
│   │   └── report.py          # GET /api/v1/reports, /reports/{id}
│   ├── core/
│   │   ├── agent.py           # LangGraph pipeline + distance estimation
│   │   ├── calculator.py      # CO2e calculation + ESG scoring
│   │   ├── extractor.py       # Vision extraction + document classification
│   │   ├── merger.py          # Multi-document merge + conflict detection
│   │   └── rag.py             # Emission factor lookup (EPA hardcoded + ChromaDB fallback)
│   ├── knowledge_base/
│   │   ├── ingest.py          # One-time EPA data ingestion into ChromaDB
│   │   └── raw/               # Source EPA/GLEC Excel files
│   └── models/
│       └── schemas.py         # Pydantic models
├── frontend/
│   ├── app.html               # Primary frontend (single file, no framework)
│   └── app.py                 # Gradio UI (alternative, not actively maintained)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Supported document types

| Type | Used for |
|------|----------|
| Commercial Invoice (CI) | Transport info, cargo weight |
| Packing List (PL) | Net weight, packaging materials and tare weights |
| Bill of Lading / Air Waybill (BL/AWB) | Transport mode, ports, multi-leg routing stops |
| Technical Data Sheet (TDS) | Packaging material type, recycled content |
| PPWR Declaration | EU packaging compliance, disposal method |

---

## Known limitations

- Air WtW ratio (1.230) uses gasoline as proxy for Jet A — not listed separately in GLEC v3.2 Table p.79
- Scoring thresholds (200–10,000 kg/ton) are not from a published standard — calibrated to actual sea/air intensity range
- Distance confidence is 0.60 when estimated vs 1.0 when extracted from document
- Nominatim geocoding fallback requires network access
- ChromaDB RAG is effectively bypassed — packaging uses hardcoded EPA Table 9 directly, transport falls back to hardcoded on ChromaDB miss