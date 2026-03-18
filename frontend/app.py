"""
frontend/app.py

Gradio UI cho Green Clearance.
Goi FastAPI backend qua httpx thay vi import truc tiep
de giu separation of concerns — UI khong biet gi ve pipeline internals.

Chay: python -m frontend.app
FastAPI phai dang chay tren port 8000 truoc.
"""

import httpx
import gradio as gr

BACKEND_URL = "http://localhost:8000/api/v1"

LANE_COLORS = {
    "GREEN":  "#22c55e",
    "YELLOW": "#f59e0b",
    "RED":    "#ef4444",
}

LANE_LABELS = {
    "GREEN":  "Xanh — Dat chuan ESG",
    "YELLOW": "Vang — Can cai thien",
    "RED":    "Do — Vuot nguong phat thai",
}


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def process_document(file_path: str, document_type: str) -> tuple:
    """
    Nhan duong dan file tu Gradio, gui len FastAPI, parse ket qua.
    Tra ve tuple tuong ung voi cac output components.
    """
    if file_path is None:
        return _empty_state("Vui long upload file chung tu.")

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        filename = file_path.split("/")[-1]
        content_type = _guess_content_type(filename)

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{BACKEND_URL}/upload",
                files={"file": (filename, file_bytes, content_type)},
                data={"document_type": document_type},
            )

        if response.status_code != 200:
            detail = response.json().get("detail", "Loi khong xac dinh.")
            return _empty_state(f"Loi {response.status_code}: {detail}")

        data = response.json()
        return _parse_response(data)

    except httpx.ConnectError:
        return _empty_state("Khong ket duoc FastAPI backend. Kiem tra server dang chay chua.")
    except Exception as e:
        return _empty_state(f"Loi: {str(e)}")


def _guess_content_type(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    return {
        "pdf":  "application/pdf",
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "webp": "image/webp",
    }.get(ext, "application/octet-stream")


def _parse_response(data: dict) -> tuple:
    score_data = data.get("score", {})
    extracted = data.get("extracted", {})
    explanation = data.get("explanation", "")
    flags = data.get("flags", [])

    lane = score_data.get("lane", "RED")
    score = score_data.get("score", 0.0)
    total_co2e = score_data.get("total_co2e_kg", 0.0)
    transport_co2e = score_data.get("transport_co2e_kg", 0.0)
    packaging_co2e = score_data.get("packaging_co2e_kg", 0.0)

    lane_html = _lane_badge(lane, score)
    co2e_html = _co2e_breakdown(total_co2e, transport_co2e, packaging_co2e)
    extracted_html = _extracted_summary(extracted)
    flag_html = _flags_summary(flags)

    return lane_html, co2e_html, extracted_html, explanation, flag_html


def _empty_state(message: str) -> tuple:
    error_html = f'<div style="color:#ef4444;padding:12px">{message}</div>'
    return error_html, "", "", "", ""


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _lane_badge(lane: str, score: float) -> str:
    color = LANE_COLORS.get(lane, "#6b7280")
    label = LANE_LABELS.get(lane, lane)
    return f"""
    <div style="text-align:center;padding:24px 0">
        <div style="
            display:inline-block;
            background:{color};
            color:white;
            border-radius:12px;
            padding:16px 40px;
            font-size:22px;
            font-weight:600;
            letter-spacing:0.5px;
        ">{label}</div>
        <div style="margin-top:12px;font-size:36px;font-weight:700;color:{color}">
            {score:.1f} <span style="font-size:18px;color:#6b7280">/ 100</span>
        </div>
    </div>
    """


def _co2e_breakdown(total: float, transport: float, packaging: float) -> str:
    def fmt(val: float) -> str:
        if val >= 1000:
            return f"{val/1000:.2f} tCO2e"
        return f"{val:.1f} kg CO2e"

    transport_pct = (transport / total * 100) if total > 0 else 0
    packaging_pct = (packaging / total * 100) if total > 0 else 0

    return f"""
    <div style="padding:8px 0">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <span style="color:#6b7280">Tong phat thai</span>
            <span style="font-weight:600;font-size:18px">{fmt(total)}</span>
        </div>
        <div style="background:#f3f4f6;border-radius:8px;overflow:hidden;height:8px;margin-bottom:16px">
            <div style="
                height:100%;
                width:{transport_pct:.1f}%;
                background:#3b82f6;
                display:inline-block;
            "></div><div style="
                height:100%;
                width:{packaging_pct:.1f}%;
                background:#f59e0b;
                display:inline-block;
            "></div>
        </div>
        <div style="display:flex;gap:16px">
            <div style="flex:1;background:#eff6ff;border-radius:8px;padding:12px">
                <div style="color:#3b82f6;font-size:12px;margin-bottom:4px">Van chuyen</div>
                <div style="font-weight:600">{fmt(transport)}</div>
                <div style="color:#6b7280;font-size:12px">{transport_pct:.1f}%</div>
            </div>
            <div style="flex:1;background:#fffbeb;border-radius:8px;padding:12px">
                <div style="color:#f59e0b;font-size:12px;margin-bottom:4px">Dong goi</div>
                <div style="font-weight:600">{fmt(packaging)}</div>
                <div style="color:#6b7280;font-size:12px">{packaging_pct:.1f}%</div>
            </div>
        </div>
    </div>
    """


def _extracted_summary(extracted: dict) -> str:
    def val(field: dict) -> str:
        v = field.get("value")
        c = field.get("confidence", 0)
        if v is None:
            return '<span style="color:#ef4444">Khong tim thay</span>'
        conf_color = "#22c55e" if c >= 0.75 else "#f59e0b"
        return f'{v} <span style="color:{conf_color};font-size:11px">({c:.0%})</span>'

    rows = [
        ("Phuong thuc", val(extracted.get("transport_mode", {}))),
        ("Cang xuat", val(extracted.get("origin_port", {}))),
        ("Cang den", val(extracted.get("destination_port", {}))),
        ("Khoang cach", val(extracted.get("distance_km", {}))),
        ("Trong luong hang", val(extracted.get("cargo_weight_tons", {}))),
    ]

    packaging_items = extracted.get("packaging_items", [])
    for i, item in enumerate(packaging_items):
        rows.append((
            f"Dong goi {i+1}",
            f"{item.get('weight_tons', 0)}t "
            f"{item.get('material', '?')} "
            f"({item.get('disposal_method', '?')})"
        ))

    rows_html = "".join(
        f"""<tr>
            <td style="padding:8px 12px;color:#6b7280;font-size:13px">{label}</td>
            <td style="padding:8px 12px;font-size:13px">{value}</td>
        </tr>"""
        for label, value in rows
    )

    return f"""
    <table style="width:100%;border-collapse:collapse">
        <tbody>{rows_html}</tbody>
    </table>
    """


def _flags_summary(flags: list[str]) -> str:
    if not flags:
        return '<div style="color:#22c55e;font-size:13px">Tat ca truong du lieu deu co do tin cay cao.</div>'

    items = "".join(
        f'<li style="margin-bottom:4px;font-size:13px">{f}</li>'
        for f in flags
    )
    return f"""
    <div style="color:#f59e0b;margin-bottom:8px;font-weight:500">
        {len(flags)} truong can kiem tra lai:
    </div>
    <ul style="margin:0;padding-left:20px;color:#6b7280">{items}</ul>
    """


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Green Clearance") as demo:
    gr.Markdown("## Green Clearance — ESG Scoring cho chung tu logistics")
    gr.Markdown("Upload hoa don / packing list / bill of lading de tinh diem ESG tu dong.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload chung tu",
                file_types=[".pdf", ".jpg", ".jpeg", ".png", ".webp"],
            )
            doc_type = gr.Dropdown(
                choices=["invoice", "packing_list", "bill_of_lading"],
                value="invoice",
                label="Loai chung tu",
            )
            submit_btn = gr.Button("Phan tich", variant="primary")

        with gr.Column(scale=2):
            lane_output = gr.HTML(label="Ket qua ESG")
            co2e_output = gr.HTML(label="Phat thai CO2e")

    with gr.Row():
        with gr.Column():
            extracted_output = gr.HTML(label="Thong tin trich xuat")
        with gr.Column():
            explanation_output = gr.Textbox(
                label="Giai thich",
                lines=5,
                interactive=False,
            )

    flags_output = gr.HTML(label="Canh bao")

    submit_btn.click(
        fn=process_document,
        inputs=[file_input, doc_type],
        outputs=[lane_output, co2e_output, extracted_output, explanation_output, flags_output],
    )


if __name__ == "__main__":
    demo.launch(server_port=7860)