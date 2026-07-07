"""Adapter: bridges the Phase 1 OCR pipeline to the agentic layer's schemas.

Why an adapter exists: the ingestion layer speaks "raw extraction" (dates in
whatever format the receipt used, floats for money). The agentic layer speaks
"validated schema" (ISO dates, decimal strings). The adapter is the single
place where translation happens, so a format change in either layer touches
one file. Alternative considered: make extract_date() return ISO directly.
Rejected: Phase 1 outputs feed other consumers (Postgres loader, dbt models)
that expect current behavior.
"""

import re
from datetime import date
from typing import Optional

from src.ingestion.ocr_pipeline import (extract_date, extract_invoice_number,
                                        extract_total, extract_vendor_name)

# SROIE is Malaysian: day-first is the correct assumption for ambiguous
# dates like 05-03-2018. Make the locale assumption explicit.

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}


def normalize_date_to_iso(raw: Optional[str]) -> Optional[str]:
    """Convert Phase 1 date formats to ISO YYYY-MM-DD. Returns None if the
    date can't be parsed confidently -- None routes the field to the LLM
    fallback, which is the correct failure mode (never guess)."""
    if not raw:
        return None
    raw = raw.strip().lower()

    # numeric: dd-mm-yyyy / dd/mm/yyyy / dd-mm-yy (day-first)
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", raw)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        # If "day" can't be a day but "month" can, receipt was month-first
        if d > 31 or (mo > 12 and d <= 12):
            d, mo = mo, d
        return _safe_iso(y, mo, d)

    # numeric: yyyy-mm-dd (already ISO-ish)
    m = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", raw)
    if m:
        return _safe_iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # textual: '5 mar 2018' / '5 march 2018'
    m = re.match(r"^(\d{1,2})\s+([a-z]+)\s+(\d{2,4})$", raw)
    if m:
        mo = _MONTHS.get(m.group(2)[:3])
        if mo:
            y = int(m.group(3))
            if y < 100:
                y += 2000
            return _safe_iso(y, mo, int(m.group(1)))

    return None


def _safe_iso(y: int, mo: int, d: int) -> Optional[str]:
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def rules_extract_fields(texts: list[str]) -> dict[str, Optional[str]]:
    """Run all Phase 1 extractors and return schema-ready values.
    None means 'rules could not extract', which the validator turns
    into an LLM-fallback ticket."""
    total = extract_total(texts)
    return {
        "vendor": extract_vendor_name(texts),
        "receipt_date": normalize_date_to_iso(extract_date(texts)),
        "total": f"{total:.2f}" if total is not None else None,
        "invoice_number": extract_invoice_number(texts),
    }
