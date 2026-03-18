from pydantic import BaseModel, Field, model_validator
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TransportMode(str, Enum):
    SEA = "sea"
    AIR = "air"
    TRUCK = "truck"
    RAIL = "rail"
    UNKNOWN = "unknown"


class PackagingMaterial(str, Enum):
    CARTON = "carton"
    HDPE = "hdpe"
    PET = "pet"
    MIXED_PLASTICS = "mixed_plastics"
    MIXED_METALS = "mixed_metals"
    STEEL = "steel"
    ALUMINUM = "aluminum"
    GLASS = "glass"
    UNKNOWN = "unknown"


class DisposalMethod(str, Enum):
    RECYCLED = "recycled"
    LANDFILLED = "landfilled"
    UNKNOWN = "unknown"


class ESGLane(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class DocumentType(str, Enum):
    """
    Cac loai chung tu logistics duoc ho tro.

    CI  — Commercial Invoice: nguon chinh cho cargo weight, origin, destination.
    PL  — Packing List: chi tiet dong goi, so kien, vat lieu.
    BL  — Bill of Lading: nguon chinh cho transport mode, ports, distance.
    TDS — Technical Data Sheet: thong tin ky thuat hang hoa, optional.
    PPWR_DOC — EU PPWR Declaration of Conformity: tuyen bo tuan thu bao bi, optional.
    UNKNOWN — Khong the classify.

    Thu tu priority khi merge conflict: BL > CI > PL > TDS > PPWR_DOC
    Ly do: BL la chung tu phap ly cao nhat cua lo hang,
    CI la nguon chinh thuc ve gia tri va trong luong,
    PL la chi tiet phu tro, TDS/PPWR la optional supplementary.
    """
    CI = "commercial_invoice"
    PL = "packing_list"
    BL = "bill_of_lading"
    TDS = "technical_data_sheet"
    PPWR_DOC = "ppwr_doc"
    UNKNOWN = "unknown"


# Document priority cho merge conflict resolution (thap = uu tien cao hon)
DOCUMENT_PRIORITY: dict[DocumentType, int] = {
    DocumentType.BL:       1,
    DocumentType.CI:       2,
    DocumentType.PL:       3,
    DocumentType.TDS:      4,
    DocumentType.PPWR_DOC: 5,
    DocumentType.UNKNOWN:  99,
}


# ---------------------------------------------------------------------------
# Extracted Document
# ---------------------------------------------------------------------------

class FieldConfidence(BaseModel):
    """
    Wrap gia tri va confidence score cua tung field.
    Field co confidence < 0.75 se bi flag de human review.
    """
    value: Optional[str | float] = None
    confidence: float = Field(ge=0.0, le=1.0)


class PackagingItem(BaseModel):
    """
    Mot loai vat lieu dong goi trong lo hang.
    Calculator se loop qua tung item va cong tong CO2e lai.
    """
    material: PackagingMaterial
    disposal_method: DisposalMethod = DisposalMethod.UNKNOWN
    weight_tons: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedDocument(BaseModel):
    transport_mode: FieldConfidence
    origin_port: FieldConfidence
    destination_port: FieldConfidence

    # distance_km co the null neu chung tu khong ghi ro.
    # LangGraph agent se goi SeaRoutes API neu null.
    distance_km: FieldConfidence

    cargo_weight_tons: FieldConfidence

    # Packaging la list vi mot lo hang thuong dung nhieu loai vat lieu.
    # Neu LLM khong tim thay, de list rong — calculator se flag "packaging_missing".
    packaging_items: list[PackagingItem] = Field(default_factory=list)

    # routing_stops: danh sach cac diem trung chuyen trich xuat tu AWB.
    # Vi du: ["SGN", "HKG", "FRA"] cho route SGN -> HKG -> FRA.
    # Neu chi co origin va destination (direct flight/sea), list se rong.
    # Agent se dung list nay de tinh tong distance theo tung leg thay vi
    # great-circle thang tu origin -> destination.
    routing_stops: list[str] = Field(default_factory=list)

    raw_text: Optional[str] = None

    @model_validator(mode="after")
    def collect_flags(self) -> "ExtractedDocument":
        """
        Flag logic:
        - value = null: luon flag — data thieu, pipeline khong the tinh duoc
        - value co nhung confidence < 0.75: flag de warn user data co the sai
        - Ngoai le: distance_km duoc estimate tu Haversine (confidence=0.60) —
          da co value roi, chi warn qua score_meta tren UI, khong flag
          vi flag se confuse user (hien "Khong tim thay" nhung thuc ra da co)
        """
        threshold = 0.75

        # Fields luon flag khi null hoac confidence thap
        strict_fields = ["transport_mode", "origin_port", "destination_port", "cargo_weight_tons"]

        # Fields chi flag khi NULL — neu co value thi chap nhan du confidence thap
        # (Haversine estimate co confidence 0.60 nhung van co gia tri su dung duoc)
        null_only_fields = ["distance_km"]

        flags = []

        for f in strict_fields:
            fc = getattr(self, f)
            if fc.value is None or fc.confidence < threshold:
                flags.append(f)

        for f in null_only_fields:
            fc = getattr(self, f)
            if fc.value is None:
                flags.append(f)

        for i, item in enumerate(self.packaging_items):
            if item.confidence < threshold:
                flags.append(f"packaging_items[{i}]")
            if item.disposal_method == DisposalMethod.UNKNOWN:
                flags.append(f"packaging_items[{i}].disposal_method")
        if not self.packaging_items:
            flags.append("packaging_items_missing")

        self.low_confidence_fields = flags
        return self

    low_confidence_fields: list[str] = Field(default_factory=list)


class SubStep(BaseModel):
    """
    Mot buoc nho trong qua trinh tinh toan — breakdown cong thuc cho UI.
    Transport step co 3 sub-steps:
        1. Unit conversion: EPA short ton-mile -> metric ton-km (TTW)
        2. WtW correction: TTW factor x WtW/TTW ratio
        3. Final: cargo_tons x distance_km x wtw_factor
    Packaging step co 2 sub-steps:
        1. Unit conversion: EPA metric ton CO2e / short ton -> / metric ton
        2. Final: weight_tons x factor x 1000 (kg)
    """
    label: str     # "Step 1 — Unit conversion"
    formula: str   # "0.0439 ÷ 1.4598"
    result: str    # "= 0.03007 kg CO2 / metric ton-km (TTW)"
    note: str = "" # "EPA GHG Hub 2025 Table 8 | 1 short ton = 0.907 MT, 1 mile = 1.609 km"


class CalculationStep(BaseModel):
    """Chi tiet mot buoc tinh toan CO2e."""
    label: str
    factor_key: str
    factor_value: float
    factor_unit: str
    quantity: float
    quantity_unit: str
    distance_km: float | None
    co2e_kg: float
    source: str
    sub_steps: list[SubStep] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ESG Score
# ---------------------------------------------------------------------------

class ESGScore(BaseModel):
    transport_co2e_kg: float = Field(ge=0.0)
    packaging_co2e_kg: float = Field(ge=0.0)
    total_co2e_kg: float = Field(ge=0.0)
    calculation_steps: list[CalculationStep] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=100.0)
    lane: ESGLane
    emission_factors_used: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_total(self) -> "ESGScore":
        expected = self.transport_co2e_kg + self.packaging_co2e_kg
        if abs(self.total_co2e_kg - expected) > 0.01:
            raise ValueError(
                f"total_co2e_kg ({self.total_co2e_kg}) "
                f"khong khop transport + packaging ({expected:.2f})"
            )
        return self


# ---------------------------------------------------------------------------
# Report Response (single document)
# ---------------------------------------------------------------------------

class ReportResponse(BaseModel):
    extracted: ExtractedDocument
    score: ESGScore
    explanation: str
    flags: list[str] = Field(default_factory=list)
    needs_human_review: bool = False

    @model_validator(mode="after")
    def sync_flags(self) -> "ReportResponse":
        self.flags = self.extracted.low_confidence_fields
        self.needs_human_review = len(self.flags) > 0
        return self


# ---------------------------------------------------------------------------
# Multi-Document Models
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """
    Ket qua classify mot file don le.

    doc_type: loai chung tu duoc phan loai.
    confidence: do tin cay cua classification.
    method: "heuristic" hoac "llm" — de trace sau.
    matched_keywords: cac tu khoa da match (chi co khi method=heuristic).
    """
    doc_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = Field(default="heuristic")
    matched_keywords: list[str] = Field(default_factory=list)


class FileAnalysis(BaseModel):
    """
    Ket qua phan tich mot file trong batch multi-document upload.

    filename: ten file goc.
    doc_type: loai chung tu sau khi classify.
    classification_confidence: do tin cay cua classification.
    classification_method: "heuristic" hoac "llm".
    extracted: data trich xuat tu file nay.
    error: neu co loi trong qua trinh xu ly file nay (khong crash toan bo batch).
    """
    filename: str
    doc_type: DocumentType
    classification_confidence: float = Field(ge=0.0, le=1.0)
    classification_method: str = Field(default="heuristic")
    extracted: Optional[ExtractedDocument] = None
    error: Optional[str] = None


class ConflictSeverity(str, Enum):
    """
    Muc do nghiem trong cua conflict.

    CRITICAL: chenh lech > 10% tren field anh huong truc tiep den CO2e score.
              Bat buoc human review truoc khi su dung ket qua.
    WARNING:  chenh lech 5-10% hoac field phu tro.
              Nen xem lai nhung khong block ket qua.
    INFO:     chenh lech < 5% hoac chi la khac format (e.g. "HCMC" vs "Ho Chi Minh City").
              Chi de thong bao, khong can action.
    """
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class FieldConflict(BaseModel):
    """
    Mo ta mot conflict cu the giua cac file khi extract cung mot field.

    field_name: ten field bi conflict (e.g. "cargo_weight_tons").
    values_by_source: map tu filename -> gia tri extract duoc.
    resolved_value: gia tri cuoi cung sau khi resolve (tu file priority cao nhat
                    hoac confidence cao nhat neu priority bang nhau).
    resolution_source: filename cua file cung cap resolved_value.
    resolution_reason: giai thich tai sao chon file nay
                       (e.g. "BL has highest document priority").
    severity: CRITICAL / WARNING / INFO dua tren muc do chenh lech.
    """
    field_name: str
    values_by_source: dict[str, str]
    resolved_value: Optional[str]
    resolution_source: Optional[str]
    resolution_reason: str
    severity: ConflictSeverity


class MultiDocumentReportResponse(BaseModel):
    """
    Response tra ve cho /api/v1/upload/multi.

    per_file: ket qua phan tich tung file (classify + extract).
    merged: ExtractedDocument sau khi merge tat ca file, conflict da duoc resolve.
    conflicts: danh sach tat ca conflict phat hien duoc giua cac file.
    score: ESGScore tinh tren merged document.
    explanation: giai thich bang ngon ngu tu nhien do LLM generate.
    flags: tong hop flags tu merged document.
    needs_human_review: True neu co bat ky CRITICAL conflict nao hoac co flag.
    document_types_found: list DocumentType da upload (de UI hien thi summary).
    missing_recommended_types: cac loai chung tu nen co nhung chua upload.
    """
    per_file: list[FileAnalysis]
    merged: ExtractedDocument
    conflicts: list[FieldConflict] = Field(default_factory=list)
    score: ESGScore
    explanation: str
    flags: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    document_types_found: list[DocumentType] = Field(default_factory=list)
    missing_recommended_types: list[DocumentType] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_review_flag(self) -> "MultiDocumentReportResponse":
        has_critical_conflict = any(
            c.severity == ConflictSeverity.CRITICAL for c in self.conflicts
        )
        self.needs_human_review = (
            has_critical_conflict
            or len(self.flags) > 0
        )
        return self


# ---------------------------------------------------------------------------
# Upload Request (single document — keep backward compat)
# ---------------------------------------------------------------------------

class UploadRequest(BaseModel):
    filename: str
    document_type: str = Field(
        default="invoice",
        pattern="^(invoice|packing_list|bill_of_lading)$"
    )