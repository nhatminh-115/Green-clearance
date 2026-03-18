"""
core/extractor.py

Doc chung tu logistics (PDF hoac anh scan) va extract structured data.

Flow extract_document():
    1. Groq vision (llama-4-scout) doc file -> raw text
       - PDF: tu dong convert sang PNG qua pdf2image + poppler
       - Anh: gui thang
    2. Groq LLaMA 3.3 70B parse raw text -> JSON theo ExtractedDocument schema
    3. Validate JSON vao Pydantic model -> tra ve ExtractedDocument

Flow classify_document():
    1. Heuristic: keyword matching tren raw text (nhanh, khong ton token)
    2. Fallback LLM: neu heuristic confidence < 0.70, goi Groq classify
    3. Tra ve ClassificationResult voi doc_type, confidence, method
"""

import io
import json
import base64
import logging
import mimetypes
import re

from PIL import Image
from groq import Groq

from backend.config import get_settings
from backend.models.schemas import (
    ClassificationResult, DocumentType,
    ExtractedDocument, FieldConfidence, PackagingItem,
    TransportMode, PackagingMaterial, DisposalMethod,
)

log = logging.getLogger(__name__)
settings = get_settings()

_groq_client = Groq(api_key=settings.groq_api_key)

_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ---------------------------------------------------------------------------
# PDF -> PNG conversion
# ---------------------------------------------------------------------------

# DPI cho PDF -> PNG conversion.
# 220 DPI: text tren document scan ro rang hon, giam hallucination cua vision model.
# Cao hon nua (300+) lam tang image size > Groq limit ma khong cai thien nhieu.
_PDF_DPI = 220

# Max dimension de Groq xu ly on dinh.
# Groq khuyen nghi image khong qua 1568px chieu dai nhat.
_MAX_IMAGE_DIMENSION = 1568


def _resize_if_needed(img: "Image.Image") -> "Image.Image":
    """Scale down neu anh qua lon, giu aspect ratio."""
    w, h = img.size
    max_dim = max(w, h)
    if max_dim <= _MAX_IMAGE_DIMENSION:
        return img
    scale = _MAX_IMAGE_DIMENSION / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), resample=3)  # LANCZOS=3


def _pdf_to_png_bytes(pdf_bytes: bytes) -> bytes:
    from pdf2image import convert_from_bytes
    from PIL import Image

    _poppler = settings.poppler_path if settings.poppler_path else None

    images = convert_from_bytes(
        pdf_bytes,
        dpi=_PDF_DPI,
        poppler_path=_poppler, #type: ignore
    )

    if len(images) == 1:
        img = _resize_if_needed(images[0])
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    # Nhieu trang: stack doc theo chieu doc thanh 1 anh
    # Groq nhan 1 image nen gop tat ca trang vao de khong bo sot data
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height

    combined = _resize_if_needed(combined)
    buf = io.BytesIO()
    combined.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Step 1: Groq vision doc file -> raw text
# ---------------------------------------------------------------------------

_VISION_PROMPT = """
You are a logistics document reader. Extract ALL text from this document exactly as written.
Preserve table structure using | as column separator.
Do not summarize or interpret — output raw text only.
If a value is unclear or illegible, write [UNCLEAR].
"""


def _read_document_with_groq_vision(file_bytes: bytes, filename: str) -> str:
    """
    Dung Groq vision (llama-4-scout) doc file va tra ve raw text.
    PDF duoc convert sang PNG truoc vi Groq chi nhan image format.
    """
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        mime_type = "application/octet-stream"

    if mime_type == "application/pdf":
        log.debug("Converting PDF to PNG...")
        file_bytes = _pdf_to_png_bytes(file_bytes)
        mime_type = "image/png"

    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")

    response = _groq_client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                ],
            }
        ],
        max_completion_tokens=2000,
        temperature=0.0,
        seed=42,  # Tang consistency — Groq support seed tren mot so model
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Document Classification
# ---------------------------------------------------------------------------

# Keyword sets cho tung loai chung tu.
# Key insight: moi loai chung tu co "anchor terms" xuat hien gan nhu luon luon —
# du la form khac nhau hay ngon ngu khac nhau.
# Scoring: moi keyword match = +1 diem, tra ve doc_type co diem cao nhat.
# Neu co anchor keyword -> confidence boost len 0.90+.

_CLASSIFICATION_RULES: dict[DocumentType, dict] = {
    DocumentType.CI: {
        "anchors": [
            "commercial invoice", "invoice no", "invoice number",
            "invoice date", "unit price", "total amount", "seller",
            "buyer", "terms of payment", "incoterms",
        ],
        "supporting": [
            "invoice", "payment", "amount due", "tax invoice",
            "pro forma", "proforma", "lc number", "letter of credit",
        ],
        "negative": ["packing list", "bill of lading", "b/l", "technical data"],
    },
    DocumentType.PL: {
        # Anchor terms EXCLUSIVE to packing lists.
        # "weight memo" va "packing condition" la terms chi xuat hien tren PL/weight memo.
        # "tare weight" cung exclusive — BL khong ghi tare weight rieng.
        # "net weight" va "gross weight" van bi overlap nhung duoc them lai vi
        # co "tare weight" lam strong signal offset BL anchors.
        "anchors": [
            "packing list", "packing no", "packing slip", "weight memo",
            "packing condition", "tare weight",
            "carton no", "carton qty", "carton number",
            "number of packages", "no of cartons", "total cartons",
            "pkg no", "pkg qty", "cbm", "cubic meter",
            "measurement", "outer carton", "inner carton",
            "net weight", "gross weight",
        ],
        "supporting": [
            "packing", "package", "carton", "crate", "pallet",
            "total packages", "shipping mark", "item no",
            "qty per carton", "net wt", "gross wt", "n.w", "g.w",
            "grand total weight", "description of goods",
        ],
        "negative": ["commercial invoice", "bill of lading", "bill of lading no"],
    },
    DocumentType.BL: {
        "anchors": [
            "bill of lading", "b/l no", "b/l number", "bl number",
            "bill of lading number", "shipper", "consignee",
            "notify party", "vessel name", "voyage no",
            "port of loading", "port of discharge",
            "ocean bill", "house bill", "master bill",
            "place of receipt", "place of delivery",
        ],
        "supporting": [
            "freight", "container no", "seal no",
            "shipped on board", "clean on board",
            "on board date", "original", "surrender",
        ],
        "negative": ["commercial invoice", "packing list", "packing no"],
    },
    DocumentType.TDS: {
        "anchors": [
            "technical data sheet", "technical specification",
            "product specification", "material safety data",
            "msds", "sds", "cas number", "chemical composition",
            "technical data",
        ],
        "supporting": [
            "specification", "properties", "composition",
            "safety data", "hazardous", "flash point", "boiling point",
        ],
        "negative": [],
    },
    DocumentType.PPWR_DOC: {
        "anchors": [
            "ppwr", "packaging and packaging waste regulation",
            "declaration of conformity", "recycled content",
            "recyclability", "eu packaging regulation",
            "post-consumer recycled", "pcr content",
        ],
        "supporting": [
            "conformity", "compliance declaration", "packaging regulation",
            "recycled material", "compostable",
        ],
        "negative": [],
    },
}


def _heuristic_classify(raw_text: str) -> ClassificationResult:
    """
    Classify chung tu dua tren keyword matching.

    Algorithm:
    1. Lowercase + normalize text
    2. Dem anchor keywords va supporting keywords cho tung loai
    3. Ap dung negative penalty neu co negative keyword
    4. Doc_type co diem cao nhat la ket qua
    5. Confidence = f(anchor_hits, total_score, text_length)

    Tra ve confidence < 0.70 neu ambiguous -> caller se fallback sang LLM.
    """
    text_lower = raw_text.lower()
    # Normalize whitespace va mot so ky tu dac biet
    text_normalized = re.sub(r'[_\-/]', ' ', text_lower)
    text_normalized = re.sub(r'\s+', ' ', text_normalized)

    scores: dict[DocumentType, float] = {}
    matched_by_type: dict[DocumentType, list[str]] = {}

    for doc_type, rules in _CLASSIFICATION_RULES.items():
        anchor_hits = [kw for kw in rules["anchors"] if kw in text_normalized]
        supporting_hits = [kw for kw in rules["supporting"] if kw in text_normalized]
        negative_hits = [kw for kw in rules["negative"] if kw in text_normalized]

        # Anchor mang nhieu trong so hon supporting
        score = len(anchor_hits) * 3.0 + len(supporting_hits) * 1.0
        # Negative keyword tru diem nhung khong ve am (loai khac co the co chung)
        score = max(0.0, score - len(negative_hits) * 2.0)

        scores[doc_type] = score
        matched_by_type[doc_type] = anchor_hits + supporting_hits

    best_type = max(scores, key=lambda t: scores[t])
    best_score = scores[best_type]

    if best_score == 0.0:
        return ClassificationResult(
            doc_type=DocumentType.UNKNOWN,
            confidence=0.0,
            method="heuristic",
            matched_keywords=[],
        )

    # Tinh confidence dua tren:
    # - So luong anchor hits so voi tong anchor cua loai do
    # - Khoang cach giua best va second best (de tranh ambiguous case)
    total_anchors = len(_CLASSIFICATION_RULES[best_type]["anchors"])
    anchor_hits_count = sum(
        1 for kw in _CLASSIFICATION_RULES[best_type]["anchors"]
        if kw in text_normalized
    )

    # Sort scores de tinh separation
    sorted_scores = sorted(scores.values(), reverse=True)
    second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    separation = (best_score - second_best) / max(best_score, 1.0)

    # Anchor ratio: bao nhieu % anchor keywords duoc tim thay
    anchor_ratio = anchor_hits_count / max(total_anchors, 1)

    # Confidence formula: combo anchor_ratio va separation
    # anchor_ratio = 0.5 + separation * 0.4 + anchor_ratio * 0.1 * boost
    confidence = min(
        0.50 + (separation * 0.30) + (anchor_ratio * 0.40),
        0.97,  # Cap tai 0.97, giu lai headroom cho uncertainty
    )

    # Hard boost: neu co it nhat 2 anchor -> confidence >= 0.85
    if anchor_hits_count >= 2:
        confidence = max(confidence, 0.85)

    # Hard boost: neu co >= 1 anchor va khong co negative -> confidence >= 0.75
    if anchor_hits_count >= 1 and not any(
        kw in text_normalized
        for kw in _CLASSIFICATION_RULES[best_type]["negative"]
    ):
        confidence = max(confidence, 0.75)

    return ClassificationResult(
        doc_type=best_type,
        confidence=round(confidence, 3),
        method="heuristic",
        matched_keywords=matched_by_type[best_type][:10],  # Cap 10 de khong spam
    )


_LLM_CLASSIFY_SYSTEM = """
You are a logistics document classifier. Given document text, identify the document type.

Return ONLY valid JSON, no explanation, no markdown:
{
  "doc_type": "commercial_invoice|packing_list|bill_of_lading|technical_data_sheet|ppwr_doc|unknown",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence"
}

Document type definitions:
- commercial_invoice: seller-buyer transaction, unit prices, total amounts, payment terms
- packing_list: package details, weights, dimensions, carton numbers
- bill_of_lading: shipping contract, vessel/flight info, ports, shipper/consignee
- technical_data_sheet: product specifications, chemical/material properties
- ppwr_doc: EU packaging regulation compliance, recycled content declarations
- unknown: cannot determine from text
"""


def _llm_classify(raw_text: str) -> ClassificationResult:
    """
    Fallback classifier dung Groq LLM khi heuristic confidence thap.
    Chi goi khi heuristic tra ve confidence < 0.70.
    """
    # Truncate text de tranh ton nhieu token — 2000 chars la du de classify
    truncated = raw_text[:2000]

    try:
        response = _groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": _LLM_CLASSIFY_SYSTEM},
                {"role": "user", "content": f"Document text:\n{truncated}"},
            ],
            temperature=0.0,
            max_completion_tokens=200,
            response_format={"type": "json_object"},
        )
        raw_json = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_json)

        doc_type_str = str(parsed.get("doc_type", "unknown")).lower()
        try:
            doc_type = DocumentType(doc_type_str)
        except ValueError:
            doc_type = DocumentType.UNKNOWN

        return ClassificationResult(
            doc_type=doc_type,
            confidence=float(parsed.get("confidence", 0.5)),
            method="llm",
            matched_keywords=[],
        )

    except Exception as e:
        log.warning(f"LLM classify that bai: {e}")
        return ClassificationResult(
            doc_type=DocumentType.UNKNOWN,
            confidence=0.0,
            method="llm",
        )


def classify_document(raw_text: str) -> ClassificationResult:
    """
    Hybrid classifier: heuristic truoc, fallback LLM neu ambiguous.

    Threshold: neu heuristic confidence >= 0.70, dung luon ket qua heuristic.
    Duoi 0.70, goi LLM va chon ket qua nao co confidence cao hon.

    Args:
        raw_text: text da duoc Groq vision extract tu file.
                  Caller phai goi _read_document_with_groq_vision() truoc.

    Returns:
        ClassificationResult voi doc_type, confidence, method, matched_keywords.
    """
    heuristic_result = _heuristic_classify(raw_text)
    log.debug(
        f"Heuristic classify: {heuristic_result.doc_type} "
        f"(confidence={heuristic_result.confidence:.2f})"
    )

    _HEURISTIC_CONFIDENCE_THRESHOLD = 0.70

    if heuristic_result.confidence >= _HEURISTIC_CONFIDENCE_THRESHOLD:
        return heuristic_result

    # Heuristic khong chac chan — fallback sang LLM
    log.info(
        f"Heuristic confidence {heuristic_result.confidence:.2f} < "
        f"{_HEURISTIC_CONFIDENCE_THRESHOLD}, falling back to LLM classify"
    )
    llm_result = _llm_classify(raw_text)
    log.debug(
        f"LLM classify: {llm_result.doc_type} "
        f"(confidence={llm_result.confidence:.2f})"
    )

    # Chon ket qua co confidence cao hon
    if llm_result.confidence >= heuristic_result.confidence:
        return llm_result

    # Heuristic van tot hon LLM (e.g. LLM tra ve unknown), dung heuristic
    return heuristic_result


# ---------------------------------------------------------------------------
# Step 2: Groq parse raw text -> JSON
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Doc-type-aware extraction prompts
# ---------------------------------------------------------------------------
#
# Moi loai chung tu co layout khac nhau va cung cap cac field khac nhau.
# Dung chung mot prompt cho tat ca dan den LLM confuse cargo weight thanh
# packaging weight (BL ghi "25 tons" -> LLM nham thanh packaging).
#
# Phan cong ro rang:
#   BL  -> transport fields only (mode, ports, distance). KHONG extract packaging.
#   CI  -> transport fields + cargo_weight. KHONG extract packaging.
#   PL  -> packaging_items voi day du material + weight + disposal. Transport fields optional.
#   TDS -> packaging material type + recycled_content_pct ONLY.
#          KHONG lay weight (TDS khong ghi weight cua bao bi, chi ghi thanh phan vat lieu).
#   PPWR -> disposal_method + recycled_content_pct ONLY.
# ---------------------------------------------------------------------------

# Base schema dung chung — transport + cargo fields
_BASE_TRANSPORT_SCHEMA = """
{
  "transport_mode": {"value": "sea|air|truck|rail|unknown", "confidence": 0.0-1.0},
  "origin_port": {"value": "string or null", "confidence": 0.0-1.0},
  "destination_port": {"value": "string or null", "confidence": 0.0-1.0},
  "distance_km": {"value": number or null, "confidence": 0.0-1.0},
  "cargo_weight_tons": {"value": number or null, "confidence": 0.0-1.0},
  "packaging_items": []
}"""

_BASE_CONFIDENCE_RULES = """
Confidence rules:
- 1.0: value is explicitly stated in the document
- 0.8-0.9: value is clearly implied (e.g. "ocean freight" -> sea)
- 0.5-0.7: value is inferred from context, not directly stated
- 0.0-0.4: guessing, document is unclear or value is missing
- Set confidence to 0.0 and value to null if the field cannot be found.

Transport mode mapping:
- sea/ocean/vessel/ship/FCL/LCL -> "sea"
- air/airfreight/AWB -> "air"
- truck/road/lorry/van -> "truck"
- rail/train/railway -> "rail"
"""

# Per-doc-type system prompts
_SYSTEM_PROMPTS: dict[str, str] = {
    "bill_of_lading": f"""
You are a logistics data extractor for Bill of Lading and Air Waybill documents.
Extract transport and cargo information. Return ONLY valid JSON, no markdown.

IMPORTANT: Do NOT extract packaging_items from a Bill of Lading or Air Waybill.
The weight on a B/L or AWB is gross shipment weight, not packaging material weight.
Always return "packaging_items": [].

For Air Waybills (AWB): extract the full routing sequence from the "Routing" field.
Example: if AWB shows "SGN - HKG - FRA", extract routing_stops as ["SGN", "HKG", "FRA"].
For direct flights or sea routes with no intermediate stops, return routing_stops as [].

Output schema:
{{
  "transport_mode": {{"value": "sea|air|truck|rail|unknown", "confidence": 0.0-1.0}},
  "origin_port": {{"value": "string or null", "confidence": 0.0-1.0}},
  "destination_port": {{"value": "string or null", "confidence": 0.0-1.0}},
  "distance_km": {{"value": number or null, "confidence": 0.0-1.0}},
  "cargo_weight_tons": {{"value": number or null, "confidence": 0.0-1.0}},
  "routing_stops": ["IATA_CODE_1", "IATA_CODE_2", "..."],
  "packaging_items": []
}}

routing_stops rules:
- Include ALL stops in order including origin and destination
- Use IATA airport codes (3 letters) or port LOCODE if available
- If only direct route (no transit): return [] empty list
- Example AWB routing "SGN-HKG-FRA": ["SGN", "HKG", "FRA"]
- Example direct flight "SGN to FRA": []

{_BASE_CONFIDENCE_RULES}""",

    "commercial_invoice": f"""
You are a logistics data extractor for Commercial Invoice documents.
Extract transport and cargo information. Return ONLY valid JSON, no markdown.

IMPORTANT: Do NOT extract packaging_items from a Commercial Invoice.
Packaging weight is not reliable from invoices — it will be sourced from the Packing List.
Always return "packaging_items": [].

Output schema:{_BASE_TRANSPORT_SCHEMA}
{_BASE_CONFIDENCE_RULES}""",

    "packing_list": """
You are a logistics data extractor for Packing List and Weight Memo documents.
Extract packaging materials AND transport info. Return ONLY valid JSON, no markdown.

WEIGHT FIELD MAPPING — read carefully:
Documents often show three weight rows: TARE WEIGHT, GROSS WEIGHT, NET WEIGHT.
Map them as follows:
- "cargo_weight_tons" = NET WEIGHT (weight of goods only, excluding packaging)
  If NET WEIGHT not present, use GROSS WEIGHT as fallback.
  NEVER use TARE WEIGHT for cargo_weight_tons.
- "packaging_items[].weight_tons" = TARE WEIGHT (weight of packaging materials only:
  bags, cartons, pallets, wrapping, etc.)
  If TARE WEIGHT is not stated, set weight_tons to 0.
  TARE WEIGHT is typically much smaller than cargo weight (e.g. 0.2 MTS vs 50 MTS).

PACKING CONDITION field: use this to identify packaging material type and disposal method.
Example: "PACKED IN NEW, PP BAGS WITH PE INNER LINER" -> material: mixed_plastics

Output schema:
{
  "transport_mode": {"value": "sea|air|truck|rail|unknown", "confidence": 0.0-1.0},
  "origin_port": {"value": "string or null", "confidence": 0.0-1.0},
  "destination_port": {"value": "string or null", "confidence": 0.0-1.0},
  "distance_km": {"value": number or null, "confidence": 0.0-1.0},
  "cargo_weight_tons": {"value": number or null, "confidence": 0.0-1.0},
  "packaging_items": [
    {
      "material": "carton|hdpe|pet|mixed_plastics|mixed_metals|steel|aluminum|glass|unknown",
      "disposal_method": "recycled|landfilled|unknown",
      "weight_tons": number,
      "confidence": 0.0-1.0
    }
  ]
}

Packaging extraction rules:
- material mapping: PP bag/PE liner/plastic bag -> mixed_plastics | carton/paper box -> carton
  steel drum -> steel | aluminum foil -> aluminum | glass bottle -> glass
- If document says "new" bags/cartons (not recycled) -> disposal_method: "landfilled"
- If document says "recycled" or "eco" -> disposal_method: "recycled"
- If unclear -> disposal_method: "unknown"
- weight_tons: TARE WEIGHT value, convert MTS->tons (1:1), kg->tons (divide by 1000)

Confidence rules:
- 1.0: explicitly stated  |  0.8-0.9: clearly implied  |  0.5-0.7: inferred  |  0.0-0.4: guessing

Transport mode: sea/ocean/vessel/FCL/LCL -> sea | air/AWB -> air | truck/road -> truck | rail -> rail""",

    "technical_data_sheet": """
You are a material data extractor for Technical Data Sheets.
Extract ONLY packaging material type and recycled content information.
Return ONLY valid JSON, no markdown.

TDS documents describe product/material specifications.
Your job: identify if the material described is used as packaging,
and whether it contains recycled content.

Output schema:
{
  "transport_mode": {"value": "unknown", "confidence": 0.0},
  "origin_port": {"value": null, "confidence": 0.0},
  "destination_port": {"value": null, "confidence": 0.0},
  "distance_km": {"value": null, "confidence": 0.0},
  "cargo_weight_tons": {"value": null, "confidence": 0.0},
  "packaging_items": [
    {
      "material": "carton|hdpe|pet|mixed_plastics|mixed_metals|steel|aluminum|glass|unknown",
      "disposal_method": "recycled|landfilled|unknown",
      "weight_tons": 0,
      "confidence": 0.0-1.0,
      "recycled_content_pct": number or null
    }
  ]
}

Rules:
- weight_tons is ALWAYS 0 for TDS — weight comes from the Packing List, not TDS
- disposal_method: if recycled_content_pct > 0 -> "recycled", else -> "unknown"
- recycled_content_pct: extract percentage if stated (e.g. "30% PCR" -> 30.0)
- If this TDS does not describe a packaging material, return "packaging_items": []
- Only include ONE packaging_items entry per distinct material type""",

    "ppwr_doc": """
You are a compliance data extractor for EU PPWR (Packaging and Packaging Waste Regulation) documents.
Extract ONLY disposal method and recycled content declarations.
Return ONLY valid JSON, no markdown.

Output schema:
{
  "transport_mode": {"value": "unknown", "confidence": 0.0},
  "origin_port": {"value": null, "confidence": 0.0},
  "destination_port": {"value": null, "confidence": 0.0},
  "distance_km": {"value": null, "confidence": 0.0},
  "cargo_weight_tons": {"value": null, "confidence": 0.0},
  "packaging_items": [
    {
      "material": "carton|hdpe|pet|mixed_plastics|mixed_metals|steel|aluminum|glass|unknown",
      "disposal_method": "recycled|landfilled|unknown",
      "weight_tons": 0,
      "confidence": 0.0-1.0,
      "recycled_content_pct": number or null
    }
  ]
}

Rules:
- weight_tons is ALWAYS 0 — PPWR declarations do not contain weight of packaging materials
- disposal_method: "recycled" if document declares recycled content or recyclability compliance
- recycled_content_pct: extract the declared percentage (e.g. "minimum 30% recycled content" -> 30.0)
- confidence: 0.95 if explicitly declared in a conformity statement, 0.75 if implied
- Include one entry per packaging component declared in the document""",
}

# Default fallback prompt cho doc types khong co trong map
_DEFAULT_SYSTEM_PROMPT = f"""
You are a logistics data extractor. Parse the document text and return ONLY valid JSON.
No markdown, no explanation, no extra text — just raw JSON.

Output schema:
{{
  "transport_mode": {{"value": "sea|air|truck|rail|unknown", "confidence": 0.0-1.0}},
  "origin_port": {{"value": "string or null", "confidence": 0.0-1.0}},
  "destination_port": {{"value": "string or null", "confidence": 0.0-1.0}},
  "distance_km": {{"value": number or null, "confidence": 0.0-1.0}},
  "cargo_weight_tons": {{"value": number or null, "confidence": 0.0-1.0}},
  "packaging_items": [
    {{
      "material": "carton|hdpe|pet|mixed_plastics|mixed_metals|steel|aluminum|glass|unknown",
      "disposal_method": "recycled|landfilled|unknown",
      "weight_tons": number,
      "confidence": 0.0-1.0
    }}
  ]
}}
{_BASE_CONFIDENCE_RULES}"""


def _parse_with_groq(raw_text: str, document_type: str) -> dict:
    """
    Parse raw text voi doc-type-aware system prompt.
    BL/CI se khong extract packaging_items.
    TDS/PPWR se chi extract material + disposal + recycled_content_pct.
    PL se extract day du.
    """
    system_prompt = _SYSTEM_PROMPTS.get(document_type, _DEFAULT_SYSTEM_PROMPT)

    user_prompt = f"""Document type: {document_type}

Document text:
{raw_text}

Extract information according to your instructions and return JSON only."""

    response = _groq_client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_completion_tokens=1500,
        response_format={"type": "json_object"},
    )
    raw_json = response.choices[0].message.content or "{}"
    return json.loads(raw_json)


# ---------------------------------------------------------------------------
# Step 3: Map JSON -> Pydantic model
# ---------------------------------------------------------------------------

def _safe_field(data: dict, key: str) -> FieldConfidence:
    item = data.get(key, {})
    if not isinstance(item, dict):
        return FieldConfidence(value=None, confidence=0.0)
    return FieldConfidence(
        value=item.get("value"),
        confidence=float(item.get("confidence", 0.0)),
    )


def _safe_packaging_items(items: list) -> list[PackagingItem]:
    """
    Parse packaging items tu LLM JSON output.

    TDS va PPWR co the tra ve them field "recycled_content_pct" — field nay
    khong co trong PackagingItem schema nhung duoc dung boi merger.py de
    resolve disposal_method. Luu vao PackagingItem.recycled_content_pct
    neu schema ho tro, hoac log de merger xu ly sau.

    Logic disposal resolution:
    - Neu LLM tra ve disposal_method != unknown -> dung luon
    - Neu LLM tra ve recycled_content_pct > 0 -> set disposal = recycled
    - Nguoc lai -> unknown (se duoc override boi PPWR/TDS trong merger)
    """
    result = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            material_str = str(item.get("material", "unknown")).lower()
            disposal_str = str(item.get("disposal_method", "unknown")).lower()
            weight = float(item.get("weight_tons", 0.0))
            confidence = float(item.get("confidence", 0.0))

            # recycled_content_pct: field extra tu TDS/PPWR prompts
            recycled_pct_raw = item.get("recycled_content_pct")
            recycled_pct = float(recycled_pct_raw) if recycled_pct_raw is not None else None

            material = (
                PackagingMaterial(material_str)
                if material_str in PackagingMaterial._value2member_map_
                else PackagingMaterial.UNKNOWN
            )

            # Resolve disposal: LLM explicit > recycled_content_pct inference > unknown
            if disposal_str in DisposalMethod._value2member_map_ and disposal_str != "unknown":
                disposal = DisposalMethod(disposal_str)
            elif recycled_pct is not None and recycled_pct > 0:
                disposal = DisposalMethod.RECYCLED
                log.debug(
                    f"packaging_items[{i}]: inferred disposal=recycled "
                    f"from recycled_content_pct={recycled_pct}"
                )
            else:
                disposal = DisposalMethod.UNKNOWN

            pkg = PackagingItem(
                material=material,
                disposal_method=disposal,
                weight_tons=weight,
                confidence=confidence,
            )
            # Attach recycled_content_pct as extra attribute de merger co the dung
            # (PackagingItem la Pydantic model, dung object.__setattr__ de bypass validation)
            object.__setattr__(pkg, "_recycled_content_pct", recycled_pct)

            result.append(pkg)
        except Exception as e:
            log.warning(f"Skip packaging item [{i}] do loi parse: {e}")
    return result


def _map_to_schema(parsed: dict) -> ExtractedDocument:
    transport_raw = parsed.get("transport_mode", {})
    transport_val = str(transport_raw.get("value", "unknown")).lower()
    transport_mode = (
        TransportMode(transport_val)
        if transport_val in TransportMode._value2member_map_
        else TransportMode.UNKNOWN
    )

    # Extract routing_stops — list of IATA/LOCODE strings
    raw_stops = parsed.get("routing_stops", [])
    routing_stops = [
        str(s).strip().upper()
        for s in raw_stops
        if isinstance(s, str) and s.strip()
    ]

    return ExtractedDocument(
        transport_mode=FieldConfidence(
            value=transport_mode.value,
            confidence=float(transport_raw.get("confidence", 0.0)),
        ),
        origin_port=_safe_field(parsed, "origin_port"),
        destination_port=_safe_field(parsed, "destination_port"),
        distance_km=_safe_field(parsed, "distance_km"),
        cargo_weight_tons=_safe_field(parsed, "cargo_weight_tons"),
        packaging_items=_safe_packaging_items(parsed.get("packaging_items", [])),
        routing_stops=routing_stops,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_document(
    file_bytes: bytes,
    filename: str,
    document_type: str = "invoice",
) -> ExtractedDocument:
    """
    Entry point cho single-document pipeline (backward compatible).
    Nhan raw file bytes, tra ve ExtractedDocument.
    Ho tro ca PDF lan anh (PNG, JPEG, WEBP).
    """
    log.info(f"Extracting: {filename} (type={document_type})")

    raw_text = _read_document_with_groq_vision(file_bytes, filename)
    log.debug(f"Vision extracted {len(raw_text)} chars")

    try:
        parsed = _parse_with_groq(raw_text, document_type)
    except json.JSONDecodeError as e:
        log.error(f"Groq tra ve JSON khong hop le: {e}")
        return ExtractedDocument(
            transport_mode=FieldConfidence(value=None, confidence=0.0),
            origin_port=FieldConfidence(value=None, confidence=0.0),
            destination_port=FieldConfidence(value=None, confidence=0.0),
            distance_km=FieldConfidence(value=None, confidence=0.0),
            cargo_weight_tons=FieldConfidence(value=None, confidence=0.0),
            packaging_items=[],
            raw_text=raw_text,
        )

    doc = _map_to_schema(parsed)
    doc.raw_text = raw_text
    return doc


def extract_and_classify(
    file_bytes: bytes,
    filename: str,
) -> tuple[ClassificationResult, ExtractedDocument]:
    """
    Entry point cho multi-document pipeline.
    Chay vision mot lan, sau do classify va extract song song tren cung raw_text.
    Tra ve (ClassificationResult, ExtractedDocument).

    Ly do tach rieng khoi extract_document():
    - Multi-doc flow can doc_type tu classify() de pass vao extract
    - Tranh goi vision hai lan cho cung mot file
    """
    log.info(f"Extract + classify: {filename}")

    raw_text = _read_document_with_groq_vision(file_bytes, filename)
    log.debug(f"Vision extracted {len(raw_text)} chars for {filename}")

    # Classify dua tren raw text
    classification = classify_document(raw_text)
    log.info(
        f"Classified {filename} as {classification.doc_type} "
        f"(confidence={classification.confidence:.2f}, method={classification.method})"
    )

    # Extract dung doc_type vua classify duoc
    try:
        parsed = _parse_with_groq(raw_text, classification.doc_type.value)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse that bai cho {filename}: {e}")
        parsed = {}

    doc = _map_to_schema(parsed)
    doc.raw_text = raw_text

    return classification, doc