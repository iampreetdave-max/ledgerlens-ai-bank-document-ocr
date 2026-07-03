# LedgerLens — AI Two-Stage OCR + Vision-Language Pipeline for Bank-Document Digitisation

> Reference/demonstration implementation built for a hackathon & awards submission. Runs fully offline on synthetic data — no real bank documents, no live API keys, no client branding.

## Overview

LedgerLens turns a scanned bank check or statement page into structured, spreadsheet-ready data by combining a general-purpose OCR pass with a vision-language model (VLM) extraction pass, then gating every result through deterministic arithmetic and checksum validators before it is ever presented as "done."

This repository is a clean-room, de-branded reference implementation: it demonstrates the *architecture* of LedgerLens on synthetically generated sample data, so it can be shared publicly (GitHub, submission CD) without exposing any real financial data, credentials, or client branding. Every name, account number, and amount in this repo is invented for demonstration purposes.

## Design principle: AI where needed, determinism where not

Reading handwriting and printed text off a photographed or scanned document is a genuinely hard perception problem — that's where a model (OCR engine, then a vision-language model) earns its keep. But once text has been extracted, *checking that it is internally consistent* is a solved, 100% deterministic problem that has no business being left to a model's guess:

- Does the handwritten legal amount ("One thousand two hundred eighty-four and 50/100") match the printed courtesy amount ("$1,284.50")?
- Does the routing number on the MICR line pass its checksum, or does it contain a misread digit?
- Is the date on the document actually plausible?
- Does a statement's opening balance plus its transactions equal its stated closing balance?

LedgerLens draws this line explicitly, in code: `ledgerlens/ocr.py` and `ledgerlens/vlm.py` do the probabilistic extraction; `ledgerlens/validators.py` does the deterministic verification. Only fields that fail verification, or whose confidence lands below a threshold, are ever flagged for a human to look at.

## Two-stage pipeline: OCR -> VLM

```
   scanned page / PDF
          |
          v
  [ preprocessing.py ]   tight crop -> 2-4x upscale -> grayscale -> autocontrast -> Otsu binarize
          |
          v
  [ ocr.py  -- Stage 1 ]  generic OCR engine: pixels -> markdown
          |                (e.g. a cloud document-OCR API such as Mistral OCR)
          v
  [ vlm.py  -- Stage 2 ]  vision-language model: (image, OCR markdown) -> strict JSON
          |                (e.g. a Pixtral-class or Azure-hosted multimodal model)
          |                sampled N times, merged with a self-consistency vote()
          v
  [ validators.py ]       amount reconciliation, routing checksum, date sanity,
          |               statement balance reconciliation
          v
  [ pipeline.py ]         attach confidence + needs_review, export styled .xlsx
```

**Why two stages instead of one?** A plain OCR engine is fast and cheap but doesn't reason about field *semantics* — it doesn't inherently know which number on the page is the check number versus the amount. A VLM re-reading a whole page cold is more expensive and, in practice, less reliable than one that gets to cross-reference a cheap Stage-1 OCR pass as grounding context. Splitting the work lets each stage do the part it's actually good at.

**Self-consistency voting.** In production, Stage 2 is sampled multiple times at temperature > 0. `ledgerlens.vlm.vote()` takes N independent extractions and returns, per field, the majority value plus a blended confidence score (agreement-across-samples averaged with the model's own reported confidence). This buys a cheap, honest reliability signal without needing a second, more expensive model to check the first one.

## The validator / reconciliation gate

`ledgerlens/validators.py` is the "determinism where it counts" core of this package. Every function in it is a pure, model-free function with a single arithmetic or calendar responsibility:

| Function | What it checks |
|---|---|
| `words_to_number()` | Parses a handwritten-style written amount ("One thousand two hundred eighty-four and 50/100") into a float — a real words-to-number parser, not a lookup table. |
| `reconcile_amount()` | Cross-checks the printed numeric amount against the parsed written amount. |
| `validate_routing_number()` | Real ABA/MICR routing-number checksum: the standard 3-7-1 weighted mod-10 check-digit rule used by every US bank routing number. |
| `check_date_sanity()` | Flexible date parsing plus plausibility bounds (catches things like a misread century, or an impossible future date). |
| `reconcile_statement_balance()` | Verifies `opening_balance + sum(transactions) == closing_balance` on a statement. |
| `validate_item()` | Runs the applicable checks against one extracted item and stamps `needs_review` / `review_reasons`. |

None of these call a model, make a network request, or depend on anything but arithmetic and the standard library — they are the part of the system that is allowed to be boring, because boring here means trustworthy.

## Human-in-the-loop

LedgerLens is built to *route* uncertainty to a human, not to hide it. Every exported row carries its per-field confidence and a `needs_review` flag with plain-English reasons attached (failed reconciliation, implausible date, low-confidence field). The exported spreadsheet visually highlights rows needing review. The goal is that a reviewer's attention goes exactly where the automation is least trustworthy — cursive payee names and memo lines — rather than a rubber stamp on every row, or worse, silent trust in a field the model was actually unsure about.

## Package layout

```
ledgerlens/
  __init__.py       package exports
  schema.py         ExtractedItem dataclass (the OCR/VLM <-> validators contract)
  preprocessing.py  OpenCV/Pillow image cleanup, incl. a hand-implemented Otsu binarizer
  ocr.py            Stage-1 OCR interface (mock by default; real call shape included)
  vlm.py            Stage-2 VLM interface + self-consistency vote()
  validators.py     deterministic reconciliation / checksum / date validators
  pipeline.py       orchestration: PDF render -> crop -> OCR -> VLM -> vote -> validate -> .xlsx
  synth.py          synthetic sample-check generator (Pillow only, no real data)
demo.py             end-to-end runnable demo (mock mode, offline)
requirements.txt
.env.example
```

## Running the demo

```bash
pip install -r requirements.txt
python demo.py
```

No API key is required. With `MISTRAL_API_KEY` unset (the default), `ledgerlens.ocr` and `ledgerlens.vlm` both transparently fall back to representative **mock** implementations, so the entire pipeline — preprocessing, both extraction stages, self-consistency voting, every validator, and the `.xlsx` export — runs offline with zero network access and zero secrets.

To try the real-API code path, copy `.env.example` to `.env`, fill in a real key, and export it into your shell before running. Note that the endpoint constants in `ocr.py` / `vlm.py` are intentionally non-resolving placeholder hosts (`*.invalid`) that illustrate the request/response *shape* of a real integration (auth header, JSON body, error handling) — point them at your actual provider's documented endpoint before relying on that path.

Running `demo.py` prints the extracted JSON for a synthetic check, the result of each validator (routing-number checksum, statement-balance reconciliation, amount reconciliation, date sanity), whether the item was flagged `needs_review` and why, and writes a styled `demo_output.xlsx`.

## Honest accuracy note

This is a real, load-bearing claim, not boilerplate: OCR/VLM accuracy on bank documents is **not uniform across fields**, and pretending otherwise is how automated pipelines quietly corrupt financial data.

- **Printed / machine-readable fields** (check number, printed dates, the courtesy numeric-amount box, statement summary lines): **95–99%** first-pass accuracy in practice.
- **Cursive / handwritten fields** (payee name, the legal amount-in-words line, free-text memo): roughly **45–70%** first-pass accuracy, heavily dependent on handwriting legibility.

Because of that gap, LedgerLens treats human review of handwritten fields as **mandatory, not optional** — the confidence scoring and validator gate exist specifically to make it obvious which fields need that review, rather than presenting a fully-automated result as more trustworthy than it actually is.

## Tech stack

PyMuPDF (PDF rendering) · OpenCV & Pillow (image preprocessing) · a generic cloud OCR API such as Mistral OCR (Stage 1) · a generic vision-language model such as Pixtral or an Azure-hosted multimodal model (Stage 2) · openpyxl (styled spreadsheet export) · numpy.

## Author

Patel Vishnu Jayantibhai
B.E. CSE (AI-ML), NLJIET (GTU)
