"""
ledgerlens.synth
=================
Synthetic sample-data generator.

Produces a fake personal-check image, drawn entirely with Pillow, so the
whole demo can run without any real bank document, PII, or client sample
ever touching this repository. Every name, number, and account/routing
digit below is an invented placeholder built for this demo -- none of it
corresponds to any real person, account, or bank.
"""

from __future__ import annotations

import math
from datetime import date as _date
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# A syntactically-valid ABA routing number checksum (see
# ledgerlens.validators.validate_routing_number for the real check-digit
# rule). Deliberately built from an ascending placeholder digit sequence
# ("12345678" + a computed check digit) so it is obviously synthetic and
# does not resemble any real institution's assigned routing number.
_EXAMPLE_ROUTING_NUMBER = "123456780"
_EXAMPLE_ACCOUNT_NUMBER = "0000123456"


def _load_font(size: int) -> ImageFont.ImageFont:
    """
    Best-effort font loader: tries a couple of common system/bundled fonts
    and falls back to Pillow's built-in bitmap font, so this renders on any
    machine with zero extra dependencies or bundled font files.
    """
    for candidate in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_synthetic_check(
    *,
    check_no: str = "1042",
    date_str: Optional[str] = None,
    payee: str = "John Q. Sample",
    amount_numeric: float = 1284.50,
    amount_words: Optional[str] = None,
    memo: str = "Invoice 88213",
    bank_name: str = "Demo Community Bank",
    routing_number: str = _EXAMPLE_ROUTING_NUMBER,
    account_number: str = _EXAMPLE_ACCOUNT_NUMBER,
    width: int = 1600,
    height: int = 700,
) -> Image.Image:
    """
    Draw a synthetic personal-check image for demo purposes. Every value
    has a clearly-fake default (a placeholder bank name, a made-up payee,
    a round demo dollar amount) so nothing here resembles a real customer
    document. `date_str` defaults to today's date if not given, so the
    generated sample never looks stale.
    """
    if date_str is None:
        date_str = _date.today().strftime("%m/%d/%Y")
    if amount_words is None:
        amount_words = _number_to_words_currency(amount_numeric)

    img = Image.new("RGB", (width, height), color=(250, 249, 240))
    draw = ImageDraw.Draw(img)

    font_large = _load_font(30)
    font_medium = _load_font(24)
    font_small = _load_font(18)

    # Outer border.
    draw.rectangle([10, 10, width - 10, height - 10], outline=(60, 60, 90), width=3)

    # Bank name + check number (top).
    draw.text((40, 30), bank_name, font=font_large, fill=(20, 20, 60))
    draw.text((width - 220, 30), f"No. {check_no}", font=font_medium, fill=(20, 20, 60))

    # Date line.
    draw.text((width - 320, 90), "Date", font=font_small, fill=(90, 90, 90))
    draw.text((width - 260, 85), date_str, font=font_medium, fill=(10, 10, 10))
    draw.line([(width - 260, 118), (width - 40, 118)], fill=(10, 10, 10), width=2)

    # Payee line.
    draw.text((40, 170), "PAY TO THE", font=font_small, fill=(90, 90, 90))
    draw.text((40, 190), "ORDER OF", font=font_small, fill=(90, 90, 90))
    draw.text((180, 185), payee, font=font_medium, fill=(15, 15, 60))
    draw.line([(180, 225), (width - 220, 225)], fill=(10, 10, 10), width=2)

    # Courtesy numeric-amount box.
    draw.rectangle([width - 210, 170, width - 40, 215], outline=(10, 10, 10), width=2)
    draw.text((width - 200, 180), f"$ {amount_numeric:,.2f}", font=font_medium, fill=(15, 15, 60))

    # Legal amount-in-words line.
    draw.text((40, 250), f"{amount_words} DOLLARS", font=font_medium, fill=(15, 15, 60))
    draw.line([(40, 285), (width - 40, 285)], fill=(10, 10, 10), width=2)

    # Memo line.
    draw.text((40, 330), "MEMO", font=font_small, fill=(90, 90, 90))
    draw.text((110, 325), memo, font=font_medium, fill=(15, 15, 60))
    draw.line([(110, 360), (400, 360)], fill=(10, 10, 10), width=1)

    # Signature line: a procedural squiggle, not a real signature.
    draw.line([(width - 420, 360), (width - 40, 360)], fill=(10, 10, 10), width=1)
    _draw_fake_signature(draw, x0=width - 400, y0=345, x1=width - 60, y1=330)
    draw.text((width - 300, 365), "AUTHORIZED SIGNATURE (SYNTHETIC)", font=font_small, fill=(120, 120, 120))

    # MICR-style line (bottom): routing + account + check number.
    micr_font = _load_font(28)
    micr_text = f":{routing_number}: {account_number} {check_no}"
    draw.text((60, height - 80), micr_text, font=micr_font, fill=(20, 20, 20))

    return img


def _draw_fake_signature(draw: "ImageDraw.ImageDraw", x0: int, y0: int, x1: int, y1: int) -> None:
    """Draw a simple procedural squiggle standing in for a handwritten signature."""
    points = []
    steps = 40
    for i in range(steps + 1):
        t = i / steps
        x = x0 + (x1 - x0) * t
        y = y0 + math.sin(t * 6.5) * 14 + (y1 - y0) * t
        points.append((x, y))
    draw.line(points, fill=(20, 20, 90), width=2)


def _number_to_words_currency(amount: float) -> str:
    """
    Convert a dollar amount to its written form, e.g. 1284.50 ->
    "One thousand two hundred eighty-four and 50/100". Used only to build
    synthetic sample data. This is intentionally a separate, independent
    implementation from the words-to-number *parser* in
    ledgerlens.validators, so the sample generator can never accidentally
    mask a bug in the real validator logic that the pipeline depends on.
    """
    dollars = int(amount)
    cents = round((amount - dollars) * 100)
    words = _int_to_words(dollars).capitalize()
    return f"{words} and {cents:02d}/100"


_ONES_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS_WORDS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]


def _int_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + _int_to_words(-n)

    def three_digit(num: int) -> str:
        parts = []
        if num >= 100:
            parts.append(f"{_ONES_WORDS[num // 100]} hundred")
            num %= 100
        if num >= 20:
            tens_word = _TENS_WORDS[num // 10]
            if num % 10:
                parts.append(f"{tens_word}-{_ONES_WORDS[num % 10]}")
            else:
                parts.append(tens_word)
        elif num > 0:
            parts.append(_ONES_WORDS[num])
        return " ".join(parts)

    scales = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"), (1, "")]
    words = []
    remaining = n
    for scale_value, scale_name in scales:
        if remaining >= scale_value:
            count = remaining // scale_value
            remaining %= scale_value
            chunk = three_digit(count)
            words.append(f"{chunk} {scale_name}".strip())
    return " ".join(words) if words else "zero"


def generate_synthetic_statement_totals() -> dict:
    """
    Return a small synthetic set of opening-balance / transactions /
    closing-balance numbers, for demonstrating
    validators.reconcile_statement_balance without any real account data.
    """
    opening_balance = 5000.00
    transactions = [-1284.50, -42.10, 900.00, -75.00]  # checks/debits negative, deposits positive
    closing_balance = round(opening_balance + sum(transactions), 2)
    return {
        "opening_balance": opening_balance,
        "transactions": transactions,
        "closing_balance": closing_balance,
    }
