# Green Clearance

ESG scoring tool for international logistics shipments. Upload trade documents (Commercial Invoice, Packing List, Bill of Lading, TDS, PPWR), get CO2e emissions calculated and an ESG score with lane classification.

Built for the CAIEC competition.

---

## What it does

1. Reads uploaded documents via vision AI (Groq llama-4-scout)
2. Classifies each document type automatically
3. Extracts structured data — transport mode, ports, cargo weight, packaging materials
4. Merges data from multiple documents with conflict detection
5. Calculates CO2e emissions aligned with GLEC Framework v3.2
6. Outputs an ESG score (0–100) with GREEN / YELLOW / RED lane

---

## Stack

- **Backend:** FastAPI + LangGraph
- **AI/LLM:** Groq (llama-4-scout-17b for vision, llama-3.3-70b for classification)
- **Distance (sea):** searoute-py — actual sea routing graph, not straight-line
- **Distance (air):** Haversine x 1.09 (DEFRA methodology) / multi-leg from AWB routing
- **Fuzzy matching:** rapidfuzz
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

Score is based on **CO2e intensity** (kg CO2e per metric ton of cargo), not absolute emissions. This is intentional — a 50t sea shipment emitting 50 tCO2e scores higher than a 2t air shipment emitting 20 tCO2e, because intensity per ton is what matters for efficiency benchmarking.

```
intensity = total_co2e_kg / cargo_weight_tons
score = (10,000 - intensity) / (10,000 - 200) x 100
```

The thresholds (200–10,000 kg/ton) are calibrated to the actual range of logistics operations — best-case sea short haul to worst-case air long haul. This is a proprietary scoring scale, not an industry standard.

| Lane   | Score | Typical intensity |
|--------|-------|-------------------|
| GREEN  | >= 70 | < ~3,100 kg/ton |
| YELLOW | 40-69 | ~3,100–6,200 kg/ton |
| RED    | < 40  | > ~6,200 kg/ton |

### Distance estimation

When documents do not include distance:

- **Sea:** searoute-py library (actual sea routing, accounts for straits)
- **Air direct:** Haversine x 1.09 — DEFRA GHG Conversion Factors methodology
- **Air multi-leg:** Sum of per-leg Haversine x 1.02 (ATC deviation), routing extracted from AWB
- **Truck:** Haversine x 1.18–1.35 (road detour factor by distance)

### Document merge priority

When multiple documents provide the same field, priority order is:

```
BL(1) > CI(2) > PL(3) > TDS(4) > PPWR(5)
```

Exception: `cargo_weight_tons` uses **PL(1) > CI(2) > BL(3)** because BL reports gross weight (cargo + packaging), not net weight needed for emissions calculation.

Packaging weight comes from PL only. Disposal method comes from PPWR > TDS > PL.

---

## Setup

```bash
# Install dependencies
pip install fastapi uvicorn langgraph groq chromadb \
    pdf2image pillow pydantic python-multipart \
    searoute rapidfuzz httpx supabase

# Copy and fill environment variables
cp .env.example .env

# Run
uvicorn backend.main:app --reload
```

Open `frontend/app.html` in a browser (or serve via any static file server).

### Environment variables

```
GROQ_API_KEY=
SUPABASE_URL=
SUPABASE_KEY=
CHROMA_PERSIST_PATH=./chroma_db
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

---

## Project structure

```
green_clearance/
├── backend/
│   ├── api/routes/
│   │   └── upload.py          # POST /api/v1/upload, /upload/multi
│   ├── core/
│   │   ├── agent.py           # LangGraph pipeline + distance estimation
│   │   ├── calculator.py      # CO2e calculation + ESG scoring
│   │   ├── extractor.py       # Vision extraction + document classification
│   │   └── merger.py          # Multi-document merge + conflict detection
│   └── models/
│       └── schemas.py         # Pydantic models
└── frontend/
    └── app.html               # Single-file frontend
```

---

## Supported document types

| Type | Used for |
|------|----------|
| Commercial Invoice (CI) | Transport info, cargo weight |
| Packing List (PL) | Net weight, packaging materials and weights |
| Bill of Lading / Air Waybill (BL/AWB) | Transport mode, ports, AWB multi-leg routing |
| Technical Data Sheet (TDS) | Packaging material type, recycled content |
| PPWR Declaration | EU packaging compliance, disposal method |

---

## Known limitations

- Air WtW ratio (1.230) uses gasoline as proxy for Jet A — Jet A is not listed separately in GLEC v3.2 Table p.79
- Distance confidence is 0.60 when estimated (vs 1.0 when extracted from document)
- Scoring thresholds are not from a published standard — calibrated to sea/air freight intensity range
- Nominatim geocoding fallback requires network access; some environments block external domains