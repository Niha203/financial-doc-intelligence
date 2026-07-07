"""Tests for the adapter's date normalization -- runs without OCR/LLM deps
if you stub the ocr_pipeline import, or just run in your repo where
easyocr is installed. Run: pytest tests/test_adapter.py -v"""

from src.agentic.adapter import normalize_date_to_iso


def test_day_first_numeric():
    assert normalize_date_to_iso("05-03-2018") == "2018-03-05"
    assert normalize_date_to_iso("05/03/2018") == "2018-03-05"


def test_two_digit_year():
    assert normalize_date_to_iso("05-03-18") == "2018-03-05"


def test_month_first_disambiguation():
    # 03-25-2018: 25 can't be a month, so it must be month-first
    assert normalize_date_to_iso("03-25-2018") == "2018-03-25"


def test_iso_passthrough():
    assert normalize_date_to_iso("2018-03-05") == "2018-03-05"


def test_textual_month():
    assert normalize_date_to_iso("5 mar 2018") == "2018-03-05"
    assert normalize_date_to_iso("5 march 2018") == "2018-03-05"


def test_garbage_returns_none():
    assert normalize_date_to_iso("not a date") is None
    assert normalize_date_to_iso("99-99-2018") is None
    assert normalize_date_to_iso(None) is None
