import httpx
import json

def run():
    url = "http://localhost:8000/api/v1/upload/multi"
    files = [
        ("files", ("BL_mock.pdf", open("data/mock_documents/package_1/BL_mock.pdf", "rb"), "application/pdf")),
        ("files", ("CI_mock.pdf", open("data/mock_documents/package_1/CI_mock.pdf", "rb"), "application/pdf")),
        ("files", ("PL_mock.pdf", open("data/mock_documents/package_1/PL_mock.pdf", "rb"), "application/pdf")),
        ("files", ("PPWR_mock.pdf", open("data/mock_documents/package_1/PPWR_mock.pdf", "rb"), "application/pdf")),
        ("files", ("TDS_mock.pdf", open("data/mock_documents/package_1/TDS_mock.pdf", "rb"), "application/pdf")),
    ]
    with httpx.Client() as client:
        r = client.post(url, files=files, timeout=60.0)
    print("Status:", r.status_code)
    try:
        with open("multi_response.json", "w", encoding="utf-8") as f:
            json.dump(r.json(), f, indent=2)
    except Exception as e:
        print("Error parsing json:", e)
        print("Raw text:", r.text[:500])

if __name__ == "__main__":
    run()
