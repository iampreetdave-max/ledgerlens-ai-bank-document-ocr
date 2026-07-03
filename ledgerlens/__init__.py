"""
LedgerLens
==========
Demonstration package for a two-stage OCR + vision-language-model pipeline
for bank-check / statement digitisation.

Stage 1 (ledgerlens.ocr)  - general-purpose OCR: pixels -> markdown text.
Stage 2 (ledgerlens.vlm)  - vision-language extraction: (image, markdown)
                            -> structured JSON, refined via a self-
                            consistency majority vote (ledgerlens.vlm.vote).
Gate     (ledgerlens.validators) - deterministic arithmetic / checksum /
                            date validators that decide what a human must
                            review before anything is trusted.

Everything in this package runs fully offline in "mock mode" (no API key,
no network) so it can be explored, tested, and demoed with zero setup.
See demo.py at the repository root for a runnable end-to-end example on
synthetic data.
"""

from .schema import ExtractedItem
from .pipeline import run_pipeline, process_item, export_xlsx
from .synth import generate_synthetic_check

__all__ = [
    "ExtractedItem",
    "run_pipeline",
    "process_item",
    "export_xlsx",
    "generate_synthetic_check",
]

__version__ = "0.1.0"
