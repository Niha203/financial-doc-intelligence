"""Production guardrails for the extraction pipeline.

Receipts are UNTRUSTED INPUT. In a multi-tenant SaaS, anyone can upload a
document containing text designed to manipulate the LLM. Three defenses:
1. Injection detection  -- flag suspicious instruction-like text in OCR output
2. PII redaction        -- scrub before logging
3. Circuit breaker      -- hard cap on LLM calls per document
"""

import re
from dataclasses import dataclass, field

# --- 1. Prompt injection detection ---------------------------------------
# Heuristic, not perfect -- defense in depth. This flags; the real defense
# is that the LLM prompt treats OCR text strictly as data, and output is
# schema-validated.

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts)",
    r"disregard (the|your) (instructions|system prompt)",
    r"you are now",
    r"new instructions:",
    r"system prompt",
    r"</?(system|assistant|instructions?)>",
    r"do anything now",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def detect_injection(ocr_text: str) -> bool:
    """True if OCR text contains instruction-like content. We don't block
    the document -- receipts legitimately contain weird text. We flag, log,
    and the downstream prompt treats text as data only."""
    return bool(_INJECTION_RE.search(ocr_text))


# --- 2. PII redaction for logs --------------------------------------------

_PII_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"),
    "CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def redact_pii(text: str) -> str:
    for label, pattern in _PII_PATTERNS.items():
        text = pattern.sub(f"[{label}]", text)
    return text


# --- 3. Circuit breaker -----------------------------------------------------

class CircuitBreakerTripped(Exception):
    pass


@dataclass
class LLMBudget:
    """Per-document LLM call budget. Agent loops are the classic agentic
    failure mode; a hard cap converts an infinite loop into a logged,
    recoverable error."""
    max_calls: int = 3
    calls_made: int = field(default=0)

    def spend(self) -> None:
        self.calls_made += 1
        if self.calls_made > self.max_calls:
            raise CircuitBreakerTripped(
                f"LLM call budget exceeded ({self.max_calls} per document)"
            )
