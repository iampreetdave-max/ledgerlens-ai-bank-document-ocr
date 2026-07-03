"""
ledgerlens.validators
=====================
Deterministic, model-free validators. This is the "determinism where it
counts" half of LedgerLens: once the OCR/VLM stages have produced a
candidate value for a field, every check in this module answers a yes/no
question using plain arithmetic, checksums, or calendar rules -- nothing
here calls a model, makes a network request, or "guesses". These functions
are pure and unit-testable in isolation from the rest of the pipeline.

Contents:
    words_to_number             - parse a written-amount string to a float
    reconcile_amount             - numeric vs. written amount cross-check
    validate_routing_number      - real ABA/MICR routing-number checksum
    check_date_sanity            - flexible date parsing + plausibility checks
    reconcile_statement_balance  - opening + transactions == closing
    validate_item                - run the applicable checks against one
                                    ExtractedItem and stamp needs_review /
                                    review_reasons
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, List, Optional

from .schema import ExtractedItem


@dataclass
class ValidationResult:
    """Outcome of a single validator call."""
    ok: bool
    message: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Written-amount ("legal amount") parsing
# ---------------------------------------------------------------------------

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
_IGNORED_WORDS = {"and", "only", "even", "exactly", "dollars", "dollar", "the", "sum", "of"}


def _words_to_int(words_text: str) -> int:
    """Convert a run of English number words (no fraction/cents part) to an int."""
    tokens = [t for t in re.split(r"[\s\-]+", words_text.lower().strip()) if t]
    tokens = [t for t in tokens if t not in _IGNORED_WORDS]

    total = 0
    current = 0
    matched_any = False

    for tok in tokens:
        if tok in _ONES:
            current += _ONES[tok]
            matched_any = True
        elif tok in _TENS:
            current += _TENS[tok]
            matched_any = True
        elif tok == "hundred":
            current = (current or 1) * 100
            matched_any = True
        elif tok in _SCALES:
            scale = _SCALES[tok]
            total += (current or 1) * scale
            current = 0
            matched_any = True
        # Unknown tokens are ignored silently -- keeps this robust to stray
        # OCR noise words instead of raising on every minor formatting quirk.

    total += current
    if not matched_any:
        raise ValueError(f"Could not parse any number words from: {words_text!r}")
    return total


def words_to_number(text: str) -> float:
    """
    Parse a handwritten-style written amount into a float dollar value, e.g.:

        "One thousand two hundred eighty-four and 50/100"  -> 1284.50
        "Two hundred dollars and no/100"                    -> 200.0
        "Five and fifty cents"                              -> 5.50

    Supports standard English number words up to billions, a trailing
    digit-fraction cents suffix ("50/100"), the "xx/100" / "no/100"
    zero-cents idiom used on real checks, and spelled-out cents.
    """
    if not text or not text.strip():
        raise ValueError("Empty amount-in-words string")

    raw = text.strip()
    cents = 0

    frac_match = re.search(r"(\d{1,2})\s*/\s*100", raw)
    idiom_match = re.search(r"\b(xx|no)\s*/\s*100", raw, flags=re.IGNORECASE)
    cents_words_match = re.search(r"\band\s+([a-zA-Z\- ]+?)\s+cents?\b", raw, flags=re.IGNORECASE)

    if frac_match:
        cents = int(frac_match.group(1))
        raw = raw[: frac_match.start()]
    elif idiom_match:
        cents = 0
        raw = raw[: idiom_match.start()]
    elif cents_words_match:
        cents = _words_to_int(cents_words_match.group(1))
        raw = raw[: cents_words_match.start()]

    raw = re.sub(r"\band\s*$", "", raw, flags=re.IGNORECASE)
    raw = raw.strip(" \t\n.,-")

    dollars = _words_to_int(raw) if raw else 0
    return round(dollars + cents / 100.0, 2)


def reconcile_amount(
    amount_numeric: Optional[float],
    amount_words: Optional[str],
    tolerance: float = 0.01,
) -> ValidationResult:
    """
    Cross-check the printed/courtesy numeric amount against the handwritten
    legal amount-in-words. On a real check, the written amount is the
    legally-controlling one, so a mismatch here is one of the highest-value
    catches a deterministic validator can make.
    """
    if amount_numeric is None or not amount_words:
        return ValidationResult(
            ok=False,
            message="Cannot reconcile: numeric or written amount is missing.",
            details={"amount_numeric": amount_numeric, "amount_words": amount_words},
        )

    try:
        parsed_words_value = words_to_number(amount_words)
    except ValueError as exc:
        return ValidationResult(
            ok=False,
            message=f"Could not parse amount-in-words: {exc}",
            details={"amount_words": amount_words},
        )

    diff = round(abs(parsed_words_value - float(amount_numeric)), 2)
    ok = diff <= tolerance
    return ValidationResult(
        ok=ok,
        message=(
            "Numeric and written amounts agree."
            if ok
            else f"Mismatch: numeric={amount_numeric} vs. written={parsed_words_value} (diff={diff})"
        ),
        details={
            "amount_numeric": amount_numeric,
            "amount_words_parsed": parsed_words_value,
            "difference": diff,
        },
    )


# ---------------------------------------------------------------------------
# ABA / MICR routing-number checksum
# ---------------------------------------------------------------------------

def validate_routing_number(routing_number: str) -> ValidationResult:
    """
    Validate a 9-digit ABA/MICR routing number using the standard weighted
    checksum used by every US bank routing number:

        3*(d1+d4+d7) + 7*(d2+d5+d8) + 1*(d3+d6+d9)  is congruent to 0 (mod 10)

    This is the real check-digit rule (not a toy stand-in), so it correctly
    accepts well-formed routing numbers and rejects single-digit typos or
    transpositions introduced by a misread scan.
    """
    digits = re.sub(r"\D", "", routing_number or "")
    if len(digits) != 9:
        return ValidationResult(
            ok=False,
            message=f"Routing number must be exactly 9 digits, got {len(digits)}.",
            details={"input": routing_number},
        )

    d = [int(c) for c in digits]
    checksum = (
        3 * (d[0] + d[3] + d[6])
        + 7 * (d[1] + d[4] + d[7])
        + 1 * (d[2] + d[5] + d[8])
    )
    ok = (checksum % 10) == 0
    return ValidationResult(
        ok=ok,
        message=(
            "Valid ABA routing-number checksum."
            if ok
            else "Routing number fails the checksum (likely a misread digit)."
        ),
        details={"routing_number": digits, "checksum_mod_10": checksum % 10},
    )


# ---------------------------------------------------------------------------
# Date sanity
# ---------------------------------------------------------------------------

_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]


def _parse_flexible_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def check_date_sanity(
    date_str: Optional[str],
    *,
    not_before: Optional[str] = "2000-01-01",
    max_days_in_future: int = 180,
    reference_date: Optional[date] = None,
) -> ValidationResult:
    """
    Sanity-check an extracted date string. Flags: unparsable dates, dates
    absurdly far in the past (a common OCR digit-swap symptom, e.g. "1926"
    read for "2026"), and dates further post-dated than a configurable
    grace window (`max_days_in_future`).
    """
    if not date_str:
        return ValidationResult(ok=False, message="Date is missing.", details={})

    parsed = _parse_flexible_date(date_str)
    if parsed is None:
        return ValidationResult(
            ok=False,
            message=f"Could not parse date: {date_str!r}",
            details={"raw": date_str},
        )

    ref = reference_date or date.today()
    problems: List[str] = []

    if not_before is not None:
        lower_bound = _parse_flexible_date(not_before)
        if lower_bound and parsed < lower_bound:
            problems.append(
                f"date {parsed.isoformat()} is before the plausible lower bound {lower_bound.isoformat()}"
            )

    delta_days = (parsed - ref).days
    if delta_days > max_days_in_future:
        problems.append(
            f"date {parsed.isoformat()} is {delta_days} days in the future "
            f"(more than the {max_days_in_future}-day grace window)"
        )

    ok = not problems
    return ValidationResult(
        ok=ok,
        message="Date looks plausible." if ok else "; ".join(problems),
        details={"parsed_date": parsed.isoformat()},
    )


# ---------------------------------------------------------------------------
# Statement closing-balance reconciliation
# ---------------------------------------------------------------------------

def reconcile_statement_balance(
    opening_balance: float,
    transactions: Iterable[float],
    closing_balance: float,
    tolerance: float = 0.01,
) -> ValidationResult:
    """
    Verify that opening_balance + sum(transactions) == closing_balance.

    `transactions` should already be signed the way a real ledger is:
    deposits/credits positive, checks/debits negative.
    """
    tx_list = list(transactions)
    computed = round(float(opening_balance) + sum(tx_list), 2)
    diff = round(abs(computed - float(closing_balance)), 2)
    ok = diff <= tolerance
    return ValidationResult(
        ok=ok,
        message=(
            "Statement reconciles: opening balance + transactions == closing balance."
            if ok
            else (
                f"Statement does NOT reconcile: computed={computed} vs. "
                f"stated closing={closing_balance} (diff={diff})"
            )
        ),
        details={
            "opening_balance": opening_balance,
            "transaction_count": len(tx_list),
            "computed_closing": computed,
            "stated_closing": closing_balance,
            "difference": diff,
        },
    )


# ---------------------------------------------------------------------------
# Per-item validation gate
# ---------------------------------------------------------------------------

def validate_item(item: ExtractedItem, *, low_confidence_threshold: float = 0.75) -> ExtractedItem:
    """
    Run every applicable deterministic validator against one extracted item
    and stamp `needs_review` / `review_reasons` on it in place. This is the
    gate between "the AI's best guess" and "a row a downstream system or a
    human accountant can actually trust".
    """
    reasons: List[str] = []

    if item.amount_numeric is not None and item.amount_words:
        result = reconcile_amount(item.amount_numeric, item.amount_words)
        if not result.ok:
            reasons.append(f"amount reconciliation: {result.message}")

    if item.date:
        result = check_date_sanity(item.date)
        if not result.ok:
            reasons.append(f"date sanity: {result.message}")

    for field_name, conf in sorted((item.confidence or {}).items()):
        if conf < low_confidence_threshold:
            reasons.append(f"low confidence on '{field_name}' ({conf:.2f})")

    item.review_reasons = reasons
    item.needs_review = bool(reasons)
    return item
