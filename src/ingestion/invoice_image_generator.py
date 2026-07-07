"""
Invoice Image Generator
Generates realistic invoice images as PNG files using Pillow.
These images are the input to our EasyOCR pipeline.
"""

import random
import json
import csv
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
from dataclasses import dataclass

fake = Faker()
random.seed(42)
Faker.seed(42)


@dataclass
class InvoiceImageData:
    invoice_id: str
    vendor_name: str
    vendor_address: str
    invoice_number: str
    invoice_date: str
    due_date: str
    line_items: list
    subtotal: float
    tax_amount: float
    total_amount: float
    image_path: str


def generate_line_items(total_amount: float) -> list:
    """Generate realistic line items that sum to total amount."""
    n_items = random.randint(1, 5)
    items = []
    remaining = total_amount

    for i in range(n_items - 1):
        amount = round(random.uniform(
            remaining * 0.1,
            remaining * 0.6
        ), 2)
        remaining -= amount
        items.append({
            "description": fake.bs().title()[:40],
            "quantity": random.randint(1, 10),
            "unit_price": round(amount / random.randint(1, 10), 2),
            "amount": amount
        })

    items.append({
        "description": fake.bs().title()[:40],
        "quantity": 1,
        "unit_price": round(remaining, 2),
        "amount": round(remaining, 2)
    })

    return items


def create_invoice_image(
    invoice_data: InvoiceImageData,
    output_path: str,
    width: int = 800,
    height: int = 1100
) -> str:
    """Create a realistic invoice image."""

    # Create white background
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to use default font, fallback to basic
    try:
        font_large = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
        font_medium = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
    except Exception:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Colors
    dark_blue = (23, 55, 94)
    light_gray = (240, 240, 240)
    dark_gray = (80, 80, 80)
    black = (0, 0, 0)
    red = (200, 50, 50)

    # Header background
    draw.rectangle([(0, 0), (width, 120)], fill=dark_blue)

    # Company name in header
    draw.text(
        (30, 20),
        invoice_data.vendor_name[:35],
        font=font_large,
        fill=(255, 255, 255)
    )

    # INVOICE title
    draw.text(
        (600, 20),
        "INVOICE",
        font=font_large,
        fill=(255, 255, 255)
    )

    # Vendor address
    draw.text(
        (30, 55),
        invoice_data.vendor_address,
        font=font_small,
        fill=(200, 220, 255)
    )

    # Invoice details box
    draw.rectangle([(500, 55), (770, 115)], fill=(10, 40, 80))
    draw.text(
        (510, 60),
        f"Invoice #: {invoice_data.invoice_number}",
        font=font_small,
        fill=(255, 255, 255)
    )
    draw.text(
        (510, 78),
        f"Date: {invoice_data.invoice_date}",
        font=font_small,
        fill=(255, 255, 255)
    )
    draw.text(
        (510, 96),
        f"Due: {invoice_data.due_date}",
        font=font_small,
        fill=(255, 255, 255)
    )

    # Bill To section
    y = 140
    draw.text((30, y), "BILL TO:", font=font_medium, fill=dark_blue)
    draw.text(
        (30, y + 20),
        "Accounts Payable Department",
        font=font_small,
        fill=dark_gray
    )
    draw.text(
        (30, y + 36),
        "Financial Document Intelligence Corp",
        font=font_small,
        fill=dark_gray
    )
    draw.text(
        (30, y + 52),
        "123 Enterprise Blvd, Suite 400",
        font=font_small,
        fill=dark_gray
    )

    # Line items table header
    y = 240
    draw.rectangle([(30, y), (770, y + 30)], fill=dark_blue)
    draw.text((40, y + 8), "DESCRIPTION", font=font_small, fill=(255, 255, 255))
    draw.text((450, y + 8), "QTY", font=font_small, fill=(255, 255, 255))
    draw.text((530, y + 8), "UNIT PRICE", font=font_small, fill=(255, 255, 255))
    draw.text((660, y + 8), "AMOUNT", font=font_small, fill=(255, 255, 255))

    # Line items
    y = 280
    for i, item in enumerate(invoice_data.line_items):
        bg = light_gray if i % 2 == 0 else (255, 255, 255)
        draw.rectangle([(30, y), (770, y + 25)], fill=bg)
        draw.text(
            (40, y + 6),
            item["description"][:45],
            font=font_small,
            fill=black
        )
        draw.text(
            (460, y + 6),
            str(item["quantity"]),
            font=font_small,
            fill=black
        )
        draw.text(
            (535, y + 6),
            f"${item['unit_price']:,.2f}",
            font=font_small,
            fill=black
        )
        draw.text(
            (660, y + 6),
            f"${item['amount']:,.2f}",
            font=font_small,
            fill=black
        )
        y += 25

    # Totals section
    y = max(y + 20, 700)
    draw.line([(500, y), (770, y)], fill=dark_gray, width=1)

    y += 10
    draw.text((500, y), "Subtotal:", font=font_medium, fill=dark_gray)
    draw.text(
        (660, y),
        f"${invoice_data.subtotal:,.2f}",
        font=font_medium,
        fill=black
    )

    y += 25
    draw.text((500, y), "Tax (8%):", font=font_medium, fill=dark_gray)
    draw.text(
        (660, y),
        f"${invoice_data.tax_amount:,.2f}",
        font=font_medium,
        fill=black
    )

    y += 25
    draw.rectangle([(490, y), (780, y + 35)], fill=dark_blue)
    draw.text(
        (500, y + 8),
        "TOTAL DUE:",
        font=font_medium,
        fill=(255, 255, 255)
    )
    draw.text(
        (640, y + 8),
        f"${invoice_data.total_amount:,.2f}",
        font=font_medium,
        fill=(255, 255, 255)
    )

    # Payment instructions
    y += 60
    draw.text(
        (30, y),
        "PAYMENT INSTRUCTIONS",
        font=font_medium,
        fill=dark_blue
    )
    y += 20
    draw.text(
        (30, y),
        f"Please remit payment by {invoice_data.due_date}",
        font=font_small,
        fill=dark_gray
    )
    y += 16
    draw.text(
        (30, y),
        f"Reference: {invoice_data.invoice_number}",
        font=font_small,
        fill=dark_gray
    )

    # Footer
    draw.rectangle([(0, height - 40), (width, height)], fill=light_gray)
    draw.text(
        (30, height - 28),
        f"Invoice ID: {invoice_data.invoice_id} | "
        f"Generated by {invoice_data.vendor_name[:30]}",
        font=font_small,
        fill=dark_gray
    )

    # Save image
    img.save(output_path, "PNG", quality=95)
    return output_path


def generate_invoice_images(
    transactions_path: str = "data/raw/csv/transactions.csv",
    vendors_path: str = "data/raw/csv/vendors.json",
    output_dir: str = "data/raw/images",
    n_images: int = 100,
    save_ground_truth: bool = True
):
    """Generate invoice images and ground truth JSON."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    with open(vendors_path) as f:
        vendors = {v["vendor_id"]: v for v in json.load(f)}

    transactions = []
    with open(transactions_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            transactions.append(row)

    # Sample transactions
    sampled = random.sample(transactions, min(n_images, len(transactions)))

    ground_truth = []
    generated = 0

    print(f"Generating {n_images} invoice images...")

    for txn in sampled:
        vendor = vendors.get(txn["vendor_id"])
        if not vendor:
            continue

        line_items = generate_line_items(float(txn["amount_usd"]))

        invoice_data = InvoiceImageData(
            invoice_id=txn["invoice_id"],
            vendor_name=vendor["vendor_name"],
            vendor_address=(
                f"{vendor['city']}, {vendor['state']} "
                f"{vendor['zip_code']}"
            ),
            invoice_number=txn["invoice_number"],
            invoice_date=txn["invoice_date"],
            due_date=txn["due_date"],
            line_items=line_items,
            subtotal=float(txn["amount_usd"]),
            tax_amount=float(txn["tax_amount"]),
            total_amount=float(txn["total_amount"]),
            image_path=""
        )

        image_filename = f"{txn['invoice_id']}.png"
        image_path = os.path.join(output_dir, image_filename)

        create_invoice_image(invoice_data, image_path)
        invoice_data.image_path = image_path

        # Ground truth for OCR validation
        ground_truth.append({
            "invoice_id": txn["invoice_id"],
            "image_path": image_path,
            "expected_fields": {
                "vendor_name": vendor["vendor_name"],
                "invoice_number": txn["invoice_number"],
                "invoice_date": txn["invoice_date"],
                "due_date": txn["due_date"],
                "total_amount": txn["total_amount"],
                "vendor_id": txn["vendor_id"]
            }
        })

        generated += 1
        if generated % 10 == 0:
            print(f"  Generated {generated}/{n_images} images...")

    # Save ground truth
    if save_ground_truth:
        gt_path = "data/ground_truth/ocr_ground_truth.json"
        Path("data/ground_truth").mkdir(parents=True, exist_ok=True)
        with open(gt_path, "w") as f:
            json.dump(ground_truth, f, indent=2)
        print(f"Saved ground truth to {gt_path}")

    print(f"Done — {generated} invoice images saved to {output_dir}")
    return ground_truth


if __name__ == "__main__":
    generate_invoice_images(n_images=100)