"""
ledgerlens.ocr
==============
Stage-1 OCR interface.

Design: a single high-resolution crop of a check / statement line is sent
to a general-purpose OCR engine that returns *markdown* (preserving simple
layout, tables, and line breaks) rather than a flat text blob. In
production this would be a cloud document-OCR endpoint (e.g. an engine such
as Mistral OCR) reached over HTTPS with an API key supplied via the
environment.

This module never hardcodes a real key, host, or endpoint. If
MISTRAL_API_KEY is not set (the default for this public demo), run_ocr()
transparently falls back to a MOCK OCR result so the rest of the pipeline
is fully exercisable offline with zero network access and zero secrets.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

# Generic, non-identifying placeholder endpoint (note the .invalid TLD,
# which is reserved by RFC 2606 and will never resolve). Swap this for your
# real provider's documented endpoint if you wire up a live key.
OCR_API_ENDPOINT = "https://api.example-ocr-provider.invalid/v1/ocr"
OCR_API_KEY_ENV_VAR = "MISTRAL_API_KEY"


@dataclass
class OCRResult:
    """Result of the Stage-1 OCR pass over a single cropped item image."""
    markdown: str
    engine: str      # "mock" or the real provider name used
    is_mock: bool


def _image_to_data_uri(image) -> str:
    """Encode an image (PIL or numpy) as a base64 data: URI (PNG)."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mock_ocr_markdown() -> str:
    """
    Representative markdown of the kind a cloud OCR engine would return for
    a personal bank check: printed fields captured cleanly, cursive fields
    captured with lower fidelity -- reflecting real-world OCR behaviour
    rather than an unrealistically perfect transcription.
    """
    return (
        "| Field | Text |\n"
        "|---|---|\n"
        "| Check No. | 1042 |\n"
        "| Date | 07/03/2026 |\n"
        "| Pay to the order of | John Q. Sample |\n"
        "| Amount (numeric) | $1,284.50 |\n"
        "| Amount (words) | One thousand two hundred eighty-four and 50/100 |\n"
        "| Memo | Invoice 88213 |\n"
        "| Signature | (present, cursive - not transcribed) |\n"
    )


def _call_real_ocr_endpoint(image, api_key: str) -> OCRResult:
    """
    Real Stage-1 call: POST the image to a generic cloud OCR endpoint.

    Written to demonstrate the *shape* of the integration (auth header,
    JSON body, error handling) without depending on any single vendor's
    SDK. Only invoked when MISTRAL_API_KEY is present in the environment;
    the public demo never exercises this path.
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The 'requests' package is required to call a real OCR endpoint. "
            "Install it with `pip install requests`, or unset "
            f"{OCR_API_KEY_ENV_VAR} to use mock mode."
        ) from exc

    payload = {
        "model": "generic-document-ocr",
        "document": {"image_url": _image_to_data_uri(image)},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = requests.post(OCR_API_ENDPOINT, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    body = response.json()
    markdown = body.get("markdown") or body.get("text") or ""
    return OCRResult(markdown=markdown, engine="cloud-ocr", is_mock=False)


def run_ocr(image, api_key: Optional[str] = None) -> OCRResult:
    """
    Stage-1 OCR entry point.

    Parameters
    ----------
    image:
        A preprocessed crop (numpy array or PIL.Image) of a single item
        (one check, or one statement line/table region).
    api_key:
        Optional explicit key; if omitted, read from the MISTRAL_API_KEY
        environment variable. Never hardcode a key in source.

    Returns
    -------
    OCRResult
        Markdown transcription plus provenance (engine name, is_mock flag).
    """
    key = api_key if api_key is not None else os.environ.get(OCR_API_KEY_ENV_VAR)

    if not key:
        # No key configured -> demonstration / offline mode.
        return OCRResult(markdown=_mock_ocr_markdown(), engine="mock", is_mock=True)

    return _call_real_ocr_endpoint(image, key)
