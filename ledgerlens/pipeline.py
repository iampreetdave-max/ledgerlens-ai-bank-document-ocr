"""
ledgerlens.pipeline
====================
End-to-end orchestration:

    PDF page -> render -> detect/crop items -> Stage-1 OCR -> Stage-2 VLM
    (N self-consistency samples) -> majority vote -> deterministic
    validators -> styled .xlsx export

This module is deliberately thin: every real decision (image cleanup,
extraction, arithmetic checks) lives in preprocessing.py / ocr.py / vlm.py /
validators.py. pipeline.py just wires them together in the right order and
handles the PDF- and spreadsheet-specific plumbing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .ocr import run_ocr
from .preprocessing import preprocess_pipeline, tight_crop
from .schema import ExtractedItem
from .validators import validate_item
from .vlm import run_vlm, vote


def render_pdf_page(pdf_path: str, page_number: int = 0, dpi: int = 300) -> np.ndarray:
    """
    Render a single PDF page to an RGB numpy image using PyMuPDF (fitz).
    Used for statement PDFs and multi-check-per-page scan sheets.

    Imports fitz lazily so that consumers who only exercise the synthetic
    -image demo path (see demo.py) are never required to have PyMuPDF
    installed just to import this module.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number)
        zoom = dpi / 72.0  # PDF user space defaults to 72 dpi
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return np.array(image)
    finally:
        doc.close()


def detect_items(
    page_image: np.ndarray,
    layout: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[np.ndarray]:
    """
    Split a rendered page into per-item crops (one check, or one statement
    row, per crop).

    A production system would run real layout detection (table/row
    detection, or a check-specific object detector). For this demo
    pipeline, `layout` is an explicit list of (x0, y0, x1, y1) boxes; if
    omitted, the whole page is treated as a single item so the pipeline
    still runs end-to-end on arbitrary input.
    """
    if not layout:
        return [tight_crop(page_image)]
    return [tight_crop(page_image, bbox=box) for box in layout]


def process_item(
    item_image: np.ndarray,
    *,
    n_samples: int = 3,
    scale: float = 2.0,
) -> ExtractedItem:
    """
    Run the full per-item pipeline: preprocess -> Stage-1 OCR -> Stage-2
    VLM (n_samples independent draws) -> self-consistency vote ->
    validators.
    """
    # Stage-1 OCR works well on a clean, high-contrast binarized crop.
    ocr_ready = preprocess_pipeline(item_image, scale=scale, binarize=True)
    ocr_result = run_ocr(ocr_ready)

    # Stage-2 VLM gets the higher-fidelity (non-binarized) upscaled crop
    # plus the Stage-1 markdown as extra textual context -- see the note in
    # preprocessing.preprocess_pipeline for why binarize=False here.
    vlm_ready = preprocess_pipeline(item_image, scale=scale, binarize=False)

    samples = [
        run_vlm(vlm_ready, ocr_markdown=ocr_result.markdown, seed=i)
        for i in range(max(1, n_samples))
    ]
    merged = vote(samples)

    item = ExtractedItem.from_dict(merged)
    item = validate_item(item)
    return item


def run_pipeline(
    pdf_path: Optional[str] = None,
    *,
    page_image: Optional[np.ndarray] = None,
    layout: Optional[List[Tuple[int, int, int, int]]] = None,
    n_samples: int = 3,
) -> List[ExtractedItem]:
    """
    Full pipeline entry point. Provide either `pdf_path` (renders page 0
    with PyMuPDF) or an in-memory `page_image` (e.g. from ledgerlens.synth
    for the offline demo).
    """
    if page_image is None:
        if not pdf_path:
            raise ValueError("Provide either pdf_path or page_image")
        page_image = render_pdf_page(pdf_path)

    items = detect_items(page_image, layout=layout)
    return [process_item(crop, n_samples=n_samples) for crop in items]


def export_xlsx(items: List[ExtractedItem], output_path: str) -> str:
    """
    Export extracted items to a styled .xlsx workbook: one row per item, a
    header row, and highlighting for rows flagged needs_review so a
    reviewer's eye goes straight to what needs attention.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "LedgerLens Extraction"

    headers = [
        "Doc Type", "Check No.", "Date", "Payee", "Amount (numeric)",
        "Amount (words)", "Memo", "Signature Present",
        "Needs Review", "Review Reasons", "Min Confidence",
    ]
    ws.append(headers)

    header_fill = PatternFill(start_color="FF1F4E78", end_color="FF1F4E78", fill_type="solid")
    header_font = Font(color="FFFFFFFF", bold=True)
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    review_fill = PatternFill(start_color="FFFFF2CC", end_color="FFFFF2CC", fill_type="solid")
    ok_fill = PatternFill(start_color="FFE2EFDA", end_color="FFE2EFDA", fill_type="solid")

    for item in items:
        min_conf = min(item.confidence.values()) if item.confidence else 0.0
        row = [
            item.doc_type,
            item.check_no,
            item.date,
            item.payee,
            item.amount_numeric,
            item.amount_words,
            item.for_memo,
            item.signature_present,
            "YES" if item.needs_review else "no",
            "; ".join(item.review_reasons),
            round(min_conf, 2),
        ]
        ws.append(row)
        row_fill = review_fill if item.needs_review else ok_fill
        for col_idx in range(1, len(headers) + 1):
            ws.cell(row=ws.max_row, column=col_idx).fill = row_fill

    for col_idx, header in enumerate(headers, start=1):
        width = max(14, len(header) + 4)
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    ws.freeze_panes = "A2"
    wb.save(output_path)
    return output_path
