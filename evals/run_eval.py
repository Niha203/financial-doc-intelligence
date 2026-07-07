"""Evaluation harness -- reads data/ground_truth/ocr_ground_truth.json.

Modes:
    --mode rules    rules-only baseline (0 LLM calls)
    --mode hybrid   rules + LLM fallback

Hard lesson baked in: v1 of this eval reported 100% while extracting 0/30
totals, because ground-truth keys didn't match and 'no truth -> no penalty'
silently excused everything. Fixes:
  1. Loader handles the real schema (nested expected_fields)
  2. COVERAGE GUARD: reports ground-truth coverage per field and screams
     if it's low. A field with <90% coverage gets its accuracy flagged
     as untrustworthy. Score and coverage are reported together, always.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.agentic.adapter import normalize_date_to_iso
from src.agentic.graph import run_document

FIELDS = ("vendor", "receipt_date", "total", "invoice_number")


def normalize_row(row: dict) -> dict:
    """Ground truth -> canonical eval row. Handles the synthetic-invoice
    schema: fields nested under 'expected_fields'."""
    ef = row.get("expected_fields", row)  # tolerate flat rows too
    raw_date = ef.get("invoice_date") or ef.get("date") or ef.get("receipt_date")
    total = ef.get("total_amount") or ef.get("total")
    return {
        "document_id": row.get("invoice_id") or row.get("document_id")
                        or Path(str(row.get("image_path", "unknown"))).stem,
        "image_path": row.get("image_path"),
        "source": row.get("source", "synthetic"),
        "vendor": ef.get("vendor_name") or ef.get("vendor"),
        "receipt_date": normalize_date_to_iso(str(raw_date)) if raw_date else None,
        "total": f"{float(total):.2f}" if total is not None else None,
        "invoice_number": ef.get("invoice_number"),
    }


def normalize(s: str) -> str:
    return " ".join(
        "".join(c for c in s.lower() if c.isalnum() or c.isspace()).split()
    )


def vendor_match(pred: str, truth: str, threshold: float = 0.8) -> bool:
    p, t = set(normalize(pred).split()), set(normalize(truth).split())
    if not p or not t:
        return False
    return len(p & t) / len(t) >= threshold


def field_correct(name: str, pred, truth) -> bool | None:
    """True/False when judgeable; None when no ground truth exists.
    None rows are EXCLUDED from accuracy, not counted as correct."""
    if truth is None:
        return None
    if pred is None:
        return False
    pred, truth = str(pred), str(truth)
    if name == "vendor":
        return vendor_match(pred, truth)
    return normalize(pred) == normalize(truth)


def run(mode: str, gt_path: Path, limit: int | None) -> None:
    raw = json.loads(gt_path.read_text())
    rows = [normalize_row(r) for r in raw]
    rows = [r for r in rows if r["image_path"]]
    if limit:
        rows = rows[:limit]
    if not rows:
        print(f"No usable rows in {gt_path}")
        return

    max_llm = 0 if mode == "rules" else 2
    correct = defaultdict(int)
    judged = defaultdict(int)      # rows with ground truth for this field
    method_counts = defaultdict(lambda: defaultdict(int))
    total_llm_calls = 0
    failures = []

    for row in rows:
        result = run_document(row["document_id"], row["image_path"],
                              max_llm_calls=max_llm)
        total_llm_calls += result.llm_calls_used
        for name in FIELDS:
            fr = getattr(result, name)
            verdict = field_correct(name, fr.value, row[name])
            method_counts[name][fr.method.value] += 1
            if verdict is None:
                continue  # no truth -> excluded, never a free pass
            judged[name] += 1
            correct[name] += verdict
            if not verdict:
                failures.append({"doc": row["document_id"], "field": name,
                                 "pred": fr.value, "truth": row[name],
                                 "method": fr.method.value})

    n = len(rows)
    print(f"\n=== Eval [{mode}] over {n} documents ===")
    print(f"{'field':>16} | {'accuracy':>8} | {'coverage':>8} | methods")
    total_correct = total_judged = 0
    for name in FIELDS:
        cov = judged[name] / n
        acc = correct[name] / judged[name] if judged[name] else float("nan")
        flag = "  <-- LOW COVERAGE, accuracy untrustworthy" if cov < 0.9 else ""
        print(f"{name:>16} | {acc:8.1%} | {cov:8.1%} | "
              f"{dict(method_counts[name])}{flag}")
        total_correct += correct[name]
        total_judged += judged[name]
    overall = total_correct / total_judged if total_judged else float("nan")
    print(f"{'overall':>16} | {overall:8.1%} | "
          f"(judged {total_judged}/{n * len(FIELDS)} field-comparisons)")
    print(f"{'llm_calls':>16} | {total_llm_calls} total ({total_llm_calls / n:.2f}/doc)")

    out = Path(f"evals/failures_{mode}.json")
    out.write_text(json.dumps(failures, indent=2))
    print(f"\n{len(failures)} field failures -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rules", "hybrid"], default="rules")
    ap.add_argument("--ground-truth", type=Path,
                    default=Path("data/ground_truth/ocr_ground_truth.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.mode, args.ground_truth, args.limit)
