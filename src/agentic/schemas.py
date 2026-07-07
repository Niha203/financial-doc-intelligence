"""Structured output schemas for the agentic extraction pipeline.

Design notes:
- Every field carries provenance: rules or LLM fallback? Auditability in a
  multi-tenant SaaS means explaining WHY the system produced a value.
- Confidence is per-field, not per-document. A receipt can have a confident
  total and an unreadable vendor name.
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ExtractionMethod(str, Enum):
    RULES = "rules"          # deterministic pipeline (EasyOCR + heuristics)
    LLM_FALLBACK = "llm"     # LLM extracted it from raw OCR text
    NOT_FOUND = "not_found"


class FieldResult(BaseModel):
    """A single extracted field with provenance."""
    value: Optional[str] = None
    method: ExtractionMethod = ExtractionMethod.NOT_FOUND
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ReceiptExtraction(BaseModel):
    """Final structured output for one document."""
    document_id: str
    vendor: FieldResult = FieldResult()
    receipt_date: FieldResult = FieldResult()
    total: FieldResult = FieldResult()
    invoice_number: FieldResult = FieldResult()

    # pipeline metadata
    llm_calls_used: int = 0
    validation_errors: list[str] = Field(default_factory=list)
    injection_flagged: bool = False

    @property
    def complete(self) -> bool:
        return all(
            f.method != ExtractionMethod.NOT_FOUND
            for f in (self.vendor, self.receipt_date, self.total)
        )


class LLMFieldOutput(BaseModel):
    """Schema the LLM fallback must conform to. The smaller the schema,
    the fewer ways the model can go wrong."""
    vendor: Optional[str] = None
    receipt_date: Optional[str] = None  # ISO YYYY-MM-DD, validated below
    total: Optional[str] = None         # decimal string like "142.50"
    invoice_number: Optional[str] = None

    @field_validator("receipt_date")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        date.fromisoformat(v)  # hard-fail bad dates
        return v

    @field_validator("total")
    @classmethod
    def validate_total(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        cleaned = v.replace(",", "").replace("RM", "").replace("$", "").strip()
        f = float(cleaned)  # raises if not numeric
        if f < 0 or f > 1_000_000:
            raise ValueError(f"total out of plausible range: {f}")
        return f"{f:.2f}"
