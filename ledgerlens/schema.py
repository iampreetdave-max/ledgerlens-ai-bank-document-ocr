"""
ledgerlens.schema
=================
Canonical data shape for a single extracted item (one check, or one line of
a bank statement). Both the OCR/VLM stages and the validators/pipeline
stages speak this shape, so it is the contract between "AI extraction" and
"deterministic validation".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Field names that Stage-2 (VLM) extraction is expected to populate.
# Order here also drives column order in the .xlsx export.
EXTRACTED_FIELDS: List[str] = [
    "doc_type",
    "check_no",
    "date",
    "payee",
    "amount_numeric",
    "amount_words",
    "for_memo",
    "signature_present",
]


@dataclass
class ExtractedItem:
    """
    One extracted document item (typically one check, or one line of a bank
    statement) plus a per-field confidence score in [0.0, 1.0].

    Fields left as None mean "not detected / not applicable" (e.g. a
    statement summary line has no signature).
    """

    doc_type: str = "check"                # "check" | "statement_line"
    check_no: Optional[str] = None
    date: Optional[str] = None             # ISO-8601 "YYYY-MM-DD" when known
    payee: Optional[str] = None
    amount_numeric: Optional[float] = None
    amount_words: Optional[str] = None
    for_memo: Optional[str] = None
    signature_present: Optional[bool] = None

    # field_name -> confidence in [0.0, 1.0], one entry per populated field.
    confidence: Dict[str, float] = field(default_factory=dict)

    # Populated by validators.py / pipeline.py, not by the OCR/VLM stages.
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain-dict representation, convenient for json.dumps()."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedItem":
        """Build an ExtractedItem from a dict, ignoring unknown keys."""
        allowed = set(EXTRACTED_FIELDS) | {"confidence", "needs_review", "review_reasons"}
        known = {k: v for k, v in data.items() if k in allowed}
        return cls(**known)

    def confidence_for(self, field_name: str, default: float = 0.0) -> float:
        """Convenience accessor: confidence score for a single field."""
        return float(self.confidence.get(field_name, default))
