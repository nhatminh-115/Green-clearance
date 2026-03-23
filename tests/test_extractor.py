import json
from backend.core.extractor import extract_and_classify

with open("data/mock_documents/package_1/BL_mock.pdf", "rb") as f:
    pdf_bytes = f.read()

classification, extracted = extract_and_classify(pdf_bytes, "BL_mock.pdf")

print("Doc Type:", classification.doc_type)
print("Vessel Name:", extracted.vessel_name)
print("Carrier Name:", extracted.carrier_name)
print("Voyage No:", extracted.voyage_number)
print("JSON:", extracted.model_dump_json(indent=2))
