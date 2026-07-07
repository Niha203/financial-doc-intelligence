"""
OCR Pipeline
Extracts structured fields from receipt images using EasyOCR.
Handles both synthetic invoices and real scanned receipts (SROIE dataset).
Outputs structured JSON ready for PostgreSQL loading.
"""

import os
import json
import re
import easyocr
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger
import torch
from src.ingestion.preprocess import preprocess_receipt

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Initialize EasyOCR reader once — expensive operation
reader = easyocr.Reader(
    ['en'],
    gpu=False,
    verbose=False,
    recog_network='standard'
)


@dataclass
class ExtractedReceipt:
    image_path: str
    vendor_name: Optional[str]
    invoice_number: Optional[str]
    date: Optional[str]
    total_amount: Optional[float]
    tax_amount: Optional[float]
    subtotal: Optional[float]
    raw_text: list[str]
    confidence_scores: list[float]
    avg_confidence: float
    extraction_status: str  # SUCCESS, PARTIAL, FAILED


def extract_text_from_image(image_path: str) -> tuple[list, list]:
    """Run EasyOCR on image with preprocessing."""
    try:
        processed = preprocess_receipt(
            image_path,
            debug_dir="data/processed/debug"
        )
        results = reader.readtext(processed)
        texts = [r[1] for r in results]
        confidences = [r[2] for r in results]
        return texts, confidences
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return [], []


def extract_total(texts: list[str]) -> Optional[float]:
    """Extract total amount from receipt text."""

    # Normalize helper — fixes common EasyOCR misreads
    def normalize(s: str) -> str:
        return (s.lower()
                .replace('tol', 'tot')      # Tolal → Total
                .replace('totall', 'total')
                .replace('lotal', 'total')
                .replace('toial', 'total')
                .replace(',', '.'))          # comma decimals → dot

    # Strategy 1: token-by-token scan for TOTAL label
    for i, text in enumerate(texts):
        t = normalize(text.strip())

        # Skip subtotals and pre-tax lines
        if any(skip in t for skip in ['excluding', 'before', 'subtotal',
                                       'sub-total', 'discount', 'item discount']):
            continue

        if re.search(r'\btotal\b', t):
            # Try same token first
            amount_match = re.search(r'(\d+)[.,](\d{2})\s*$', t)
            if amount_match:
                try:
                    return float(f"{amount_match.group(1)}.{amount_match.group(2)}")
                except ValueError:
                    pass

            # Lookahead 1-3 tokens
            lookahead = texts[i+1:i+4]
            combined = normalize(' '.join(lookahead).strip())

            # Split token: '10 40' → 10.40
            split_match = re.match(r'^(\d+)\s+(\d{2})\b', combined)
            if split_match:
                try:
                    return float(f"{split_match.group(1)}.{split_match.group(2)}")
                except ValueError:
                    pass

            # Normal decimal: '10.40' or '10,40'
            decimal_match = re.match(r'^(?:rm\s*)?(\d+)[.,](\d{2})\b', combined)
            if decimal_match:
                try:
                    return float(f"{decimal_match.group(1)}.{decimal_match.group(2)}")
                except ValueError:
                    pass

    # Strategy 2: full text fallback patterns
    full_text = normalize('\n'.join(texts))
    fallback_patterns = [
        r'total\s*\(inclusive[^)]*\)\s*[:\s]+(?:rm\s*)?(\d+[.,]\d{2})',
        r'total\s+inclusive\s+gst[:\s]+(?:rm\s*)?(\d+[.,]\d{2})',
        r'grand\s*total[:\s]+(?:rm\s*)?(\d+[.,]\d{2})',
        r'amount\s*due[:\s]+(?:rm\s*)?(\d+[.,]\d{2})',
        r'total\s+amount[:\s]+(?:rm\s*)?(\d+[.,]\d{2})',
        r'(?:rm|myr)\s*(\d+[.,]\d{2})\s*$',
        r'cash\s+(?:rm\s*)?(\d+[.,]\d{2})',          # CASH 10.40 as last resort
    ]
    for pattern in fallback_patterns:
        match = re.search(pattern, full_text, re.MULTILINE)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except ValueError:
                continue

    return None


def extract_date(texts: list[str]) -> Optional[str]:
    """Extract date from receipt text."""
    date_patterns = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',          # 05-05-2018, 05/05/2018
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2})',            # 05-05-18
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',            # 2018-05-05
        r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})',
        r'(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{2,4})',
    ]

    full_text = '\n'.join(texts).lower()

    for pattern in date_patterns:
        match = re.search(pattern, full_text)
        if match:
            raw = match.group(1)
            # Strip trailing time if captured: '05-05-2018 13:16' → '05-05-2018'
            date_only = re.split(r'\s+\d{2}:\d{2}', raw)[0]
            return date_only.strip()

    return None

def extract_invoice_number(texts: list[str]) -> Optional[str]:
    """Extract invoice/document number from receipt text."""
    inv_patterns = [
        r'(?:inv|invoice|doc|receipt|no)[#:\s.]+([a-z0-9\-/]+)',
        r'(?:document|receipt)\s*(?:no|number|#)[:\s]+([a-z0-9\-/]+)',
        r'inv#\s*([a-z0-9\-/]+)',
    ]

    full_text = '\n'.join(texts).lower()

    for pattern in inv_patterns:
        match = re.search(pattern, full_text)
        if match:
            return match.group(1).upper()
    return None


def extract_vendor_name(texts: list[str]) -> Optional[str]:
    """Extract vendor name — usually in first few lines."""
    if not texts:
        return None

    # Vendor name is typically in the first 3 lines
    # Filter out very short strings and common non-vendor words
    skip_words = {'receipt', 'invoice', 'tax', 'date', 'time', 'tel', 'phone'}

    for text in texts[:5]:
        text_clean = text.strip()
        if (len(text_clean) > 4 and
                text_clean.lower() not in skip_words and
                not re.match(r'^\d+$', text_clean)):
            return text_clean

    return texts[0] if texts else None


def extract_tax(texts: list[str]) -> Optional[float]:
    """Extract tax amount from receipt text."""
    tax_patterns = [
        r'(?:gst|tax|vat)[:\s]+(?:rm\s*)?(\d+\.?\d*)',
        r'(?:gst|tax)\s*\(\d+%\)[:\s]+(?:rm\s*)?(\d+\.?\d*)',
    ]

    full_text = '\n'.join(texts).lower()

    for pattern in tax_patterns:
        match = re.search(pattern, full_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def process_image(image_path: str) -> ExtractedReceipt:
    """Process a single receipt image and extract structured fields."""
    logger.info(f"Processing: {image_path}")

    texts, confidences = extract_text_from_image(image_path)

    if not texts:
        return ExtractedReceipt(
            image_path=image_path,
            vendor_name=None,
            invoice_number=None,
            date=None,
            total_amount=None,
            tax_amount=None,
            subtotal=None,
            raw_text=[],
            confidence_scores=[],
            avg_confidence=0.0,
            extraction_status="FAILED"
        )

    vendor = extract_vendor_name(texts)
    invoice_num = extract_invoice_number(texts)
    date = extract_date(texts)
    total = extract_total(texts)
    tax = extract_tax(texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # Determine status
    fields_found = sum([
        vendor is not None,
        date is not None,
        total is not None
    ])

    if fields_found == 3:
        status = "SUCCESS"
    elif fields_found >= 1:
        status = "PARTIAL"
    else:
        status = "FAILED"

    return ExtractedReceipt(
        image_path=image_path,
        vendor_name=vendor,
        invoice_number=invoice_num,
        date=date,
        total_amount=total,
        tax_amount=tax,
        subtotal=total - tax if total and tax else None,
        raw_text=texts,
        confidence_scores=confidences,
        avg_confidence=round(avg_conf, 3),
        extraction_status=status
    )


def process_batch(
    image_dir: str,
    output_path: str,
    limit: Optional[int] = None
) -> list[dict]:
    """Process all images in a directory."""
    image_dir = Path(image_dir)
    images = list(image_dir.glob("*.jpg")) + \
             list(image_dir.glob("*.jpeg")) + \
             list(image_dir.glob("*.png"))

    if limit:
        images = images[:limit]

    logger.info(f"Processing {len(images)} images from {image_dir}")

    results = []
    success = partial = failed = 0

    for i, img_path in enumerate(images):
        result = process_image(str(img_path))
        results.append(asdict(result))

        if result.extraction_status == "SUCCESS":
            success += 1
        elif result.extraction_status == "PARTIAL":
            partial += 1
        else:
            failed += 1

        if (i + 1) % 10 == 0:
            logger.info(
                f"Progress: {i+1}/{len(images)} | "
                f"Success: {success} | "
                f"Partial: {partial} | "
                f"Failed: {failed}"
            )

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(
        f"Final: Success={success}, Partial={partial}, Failed={failed}"
    )
    logger.info(
        f"Success rate: {success/len(images)*100:.1f}%"
    )

    return results


if __name__ == "__main__":
    # Process real receipts — test set first (smaller, faster)
    logger.info("Starting OCR pipeline on real receipts...")

    results = process_batch(
        image_dir="data/raw/receipts/test/images",
        output_path="data/processed/ocr_extractions_test.json",
        limit=30  # Start with 30 to test
    )

    # Show sample result
    if results:
        sample = results[0]
        logger.info(f"Sample extraction:")
        logger.info(f"  Vendor: {sample['vendor_name']}")
        logger.info(f"  Date: {sample['date']}")
        logger.info(f"  Total: {sample['total_amount']}")
        logger.info(f"  Status: {sample['extraction_status']}")
        logger.info(f"  Avg confidence: {sample['avg_confidence']}")