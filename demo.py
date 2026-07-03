#!/usr/bin/env python3
"""
LedgerLens end-to-end demo (mock mode, offline, synthetic data only).

Run with:
    python demo.py

This script:
  1. Generates a synthetic check image (no real bank data).
  2. Runs it through the full LedgerLens pipeline: preprocessing ->
     Stage-1 OCR -> Stage-2 VLM (with self-consistency voting) ->
     deterministic validators.
  3. Prints the extracted JSON and validator results to the console.
  4. Also demonstrates the ABA routing-number checksum and the statement
     closing-balance reconciliation validator on synthetic numbers.
  5. Writes a styled workbook to demo_output.xlsx.

No MISTRAL_API_KEY is required: with the key unset, ledgerlens.ocr and
ledgerlens.vlm both fall back to representative MOCK implementations, so
this script has zero network dependency and uses zero secrets.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from ledgerlens.pipeline import export_xlsx, run_pipeline
from ledgerlens.synth import generate_synthetic_check, generate_synthetic_statement_totals
from ledgerlens.validators import reconcile_statement_balance, validate_routing_number


def main() -> int:
    print("=" * 72)
    print("LedgerLens demo -- synthetic data, mock OCR/VLM (no API key needed)")
    print("=" * 72)

    if os.environ.get("MISTRAL_API_KEY"):
        print("[info] MISTRAL_API_KEY detected -- pipeline will attempt real API calls.")
    else:
        print("[info] MISTRAL_API_KEY not set -- running fully offline in MOCK mode.")

    # 1. Synthetic check image ------------------------------------------------
    print("\n[1/4] Generating synthetic check image...")
    check_image = generate_synthetic_check()
    check_array = np.array(check_image)
    print(f"      -> synthetic check image: {check_array.shape[1]}x{check_array.shape[0]} px")

    # 2. Run the extraction pipeline on it -------------------------------------
    print("\n[2/4] Running preprocessing + Stage-1 OCR + Stage-2 VLM + self-consistency vote...")
    items = run_pipeline(page_image=check_array, n_samples=3)
    item = items[0]

    print("\nExtracted item (JSON):")
    print(json.dumps(item.to_dict(), indent=2, default=str))

    # 3. Extra validator demonstrations -----------------------------------------
    print("\n[3/4] Deterministic validator demonstrations:")

    routing_check = validate_routing_number("123456780")
    print(f"  - Routing-number checksum:           {'PASS' if routing_check.ok else 'FAIL'} - {routing_check.message}")

    statement = generate_synthetic_statement_totals()
    balance_check = reconcile_statement_balance(
        statement["opening_balance"], statement["transactions"], statement["closing_balance"]
    )
    print(f"  - Statement balance reconciliation:  {'PASS' if balance_check.ok else 'FAIL'} - {balance_check.message}")

    print(f"  - Item flagged needs_review:          {item.needs_review}")
    if item.review_reasons:
        for reason in item.review_reasons:
            print(f"      reason: {reason}")

    # 4. Export to a styled spreadsheet -----------------------------------------
    print("\n[4/4] Exporting demo_output.xlsx...")
    output_path = export_xlsx(items, "demo_output.xlsx")
    print(f"      -> wrote {output_path}")

    print("\nDEMO COMPLETE: synthetic check processed, validators executed, demo_output.xlsx written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
