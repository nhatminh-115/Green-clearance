import asyncio
from backend.core.agent import run_pipeline

with open("data/mock_documents/package_1/BL_mock.pdf", "rb") as f:
    pdf_bytes = f.read()

state = run_pipeline(pdf_bytes, "BL_mock.pdf", "bill_of_lading")
print("Extracted Vessel:", state["extracted"].vessel_name)
print("Vessel Eff:", state["vessel_efficiency"])
