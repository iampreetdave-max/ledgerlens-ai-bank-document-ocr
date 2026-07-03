"""
ledgerlens.vlm
==============
Stage-2 vision-language-model extraction interface.

Where Stage-1 (ledgerlens.ocr) turns pixels into raw text/markdown, Stage-2
turns a (crop, OCR-markdown) pair into *structured* JSON matching
ledgerlens.schema.ExtractedItem: it resolves ambiguous handwriting using
both the image and the OCR text as context, the same way a human reviewer
would cross-reference the printed courtesy-amount box against the
handwritten legal-amount line.

In production this would call a vision-language model such as a
Pixtral-class or an Azure-hosted GPT-4V-class multimodal endpoint, with a
JSON-schema-constrained response. This module never hardcodes a real
key/host; when MISTRAL_API_KEY is unset it uses a MOCK implementation that
returns representative structured output so the full pipeline runs offline.

Self-consistency: production runs the VLM N times (temperature > 0) and
reconciles disagreements with vote() below -- a majority vote per field.
That buys resilience against a single noisy sample without pretending the
model is more deterministic than it actually is.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from datetime import date as _date
from typing import Any, Dict, List, Optional

from .schema import EXTRACTED_FIELDS, ExtractedItem

VLM_API_KEY_ENV_VAR = "MISTRAL_API_KEY"
# Generic, non-identifying placeholder endpoint (note the .invalid TLD,
# which is reserved by RFC 2606 and will never resolve).
VLM_API_ENDPOINT = "https://api.example-vlm-provider.invalid/v1/vision-extract"


def _mock_extraction(seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Return a plausible structured extraction for the synthetic demo check,
    with small seedable jitter on the "hard" handwritten-style fields so
    repeated mock calls behave like independent noisy VLM samples. That is
    what makes vote() meaningful in the offline demo.
    """
    rng = random.Random(seed)

    # Printed/machine fields: the mock is stable and confident, mirroring
    # the very high real-world accuracy of printed-field extraction. The
    # date is computed as "today" so the demo never goes stale.
    base: Dict[str, Any] = {
        "doc_type": "check",
        "check_no": "1042",
        "date": _date.today().isoformat(),
        "amount_numeric": 1284.50,
        "signature_present": True,
    }

    # Cursive-style fields: a few candidate "reads" per field, mimicking
    # realistic first-pass VLM disagreement on handwriting.
    payee_candidates = ["John Q. Sample", "John Q. Sample", "John Q Sample"]
    words_candidates = [
        "One thousand two hundred eighty-four and 50/100",
        "One thousand two hundred eighty-four and 50/100",
        "One thousand two hundred eighty four and 50/100",
    ]
    memo_candidates = ["Invoice 88213", "Invoice 88213", "Invoice 88213 "]

    base["payee"] = rng.choice(payee_candidates)
    base["amount_words"] = rng.choice(words_candidates)
    base["for_memo"] = rng.choice(memo_candidates).strip()

    base["confidence"] = {
        "doc_type": 0.99,
        "check_no": 0.98,
        "date": 0.97,
        "amount_numeric": 0.96,
        "signature_present": 0.90,
        "payee": round(rng.uniform(0.45, 0.75), 2),
        "amount_words": round(rng.uniform(0.50, 0.80), 2),
        "for_memo": round(rng.uniform(0.55, 0.80), 2),
    }
    return base


def _call_real_vlm_endpoint(image, ocr_markdown: str, api_key: str) -> Dict[str, Any]:
    """
    Real Stage-2 call: POST the crop + Stage-1 markdown to a generic VLM
    endpoint, requesting a JSON response constrained to the extraction
    schema. Demonstrates integration shape only; not exercised in the
    public demo (no key configured).
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The 'requests' package is required to call a real VLM endpoint. "
            "Install it with `pip install requests`, or unset "
            f"{VLM_API_KEY_ENV_VAR} to use mock mode."
        ) from exc

    from .ocr import _image_to_data_uri  # reuse the same encoding helper

    payload = {
        "model": "generic-vision-language-extractor",
        "image": _image_to_data_uri(image),
        "context_markdown": ocr_markdown,
        "response_schema": {"type": "object", "properties": {f: {} for f in EXTRACTED_FIELDS}},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = requests.post(VLM_API_ENDPOINT, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def run_vlm(
    image,
    schema: type = ExtractedItem,
    ocr_markdown: str = "",
    api_key: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Stage-2 VLM entry point. Returns a dict of field -> value matching
    `schema` (default ExtractedItem), plus a nested "confidence" dict.

    Parameters
    ----------
    image : numpy.ndarray | PIL.Image
        The (preprocessed) item crop.
    schema : type
        The dataclass describing expected fields (default ExtractedItem).
    ocr_markdown : str
        Stage-1 OCR markdown, given to the VLM as extra grounding context.
    api_key : Optional[str]
        Explicit key; falls back to the MISTRAL_API_KEY environment variable.
    seed : Optional[int]
        Only used by the mock implementation, to produce reproducible
        "independent" samples for self-consistency voting in the demo.
    """
    key = api_key if api_key is not None else os.environ.get(VLM_API_KEY_ENV_VAR)

    if not key:
        result = _mock_extraction(seed=seed)
    else:
        result = _call_real_vlm_endpoint(image, ocr_markdown, key)

    # Only keep keys the schema actually declares (plus confidence) -- most
    # relevant on the real-API path, where a response could include extras.
    known_fields = set(EXTRACTED_FIELDS) | {"confidence"}
    return {k: v for k, v in result.items() if k in known_fields}


def vote(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Self-consistency majority vote across N independent run_vlm() samples.

    For each field, pick the most common value across all samples (ties
    broken by first-seen order via collections.Counter). The per-field
    confidence in the merged result blends two signals:
      - agreement ratio: fraction of samples that agreed with the winner
      - the model's own average reported confidence for that field

    That blend is a simple, honest reliability proxy: a field is only
    trusted when the model both sounds confident *and* is reproducible
    across independent samples.
    """
    if not results:
        raise ValueError("vote() requires at least one VLM result")
    if len(results) == 1:
        return dict(results[0])

    all_fields = set(EXTRACTED_FIELDS)
    for r in results:
        all_fields.update(k for k in r.keys() if k != "confidence")

    merged: Dict[str, Any] = {}
    agreement: Dict[str, float] = {}

    for field_name in all_fields:
        values = [r.get(field_name) for r in results if field_name in r]
        if not values:
            continue
        counts = Counter(values)
        winner, win_count = counts.most_common(1)[0]
        merged[field_name] = winner
        agreement[field_name] = win_count / len(values)

    blended_confidence: Dict[str, float] = {}
    for field_name, agree_ratio in agreement.items():
        reported = [
            r["confidence"][field_name]
            for r in results
            if isinstance(r.get("confidence"), dict) and field_name in r["confidence"]
        ]
        avg_reported = sum(reported) / len(reported) if reported else agree_ratio
        blended_confidence[field_name] = round((agree_ratio + avg_reported) / 2, 3)

    merged["confidence"] = blended_confidence
    return merged
