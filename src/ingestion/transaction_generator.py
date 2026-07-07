"""
Transaction Generator
Generates 50,000 synthetic invoice transactions linked to vendors.
Includes realistic fraud patterns, duplicates, and anomalies for ML training.
"""

import random
import json
import csv
from faker import Faker
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

fake = Faker()
random.seed(42)
Faker.seed(42)

INVOICE_CATEGORIES = [
    "Software License", "Professional Services", "Hardware",
    "Office Supplies", "Marketing", "Logistics", "Maintenance",
    "Consulting", "Cloud Services", "Raw Materials"
]

PAYMENT_STATUSES = ["PAID", "PENDING", "OVERDUE", "DISPUTED", "CANCELLED"]

FRAUD_PATTERNS = [
    "duplicate_invoice",
    "round_number_amount",
    "weekend_submission",
    "after_hours_submission",
    "slightly_under_approval_limit",
    "new_vendor_large_amount",
    "none"  # legitimate
]


@dataclass
class Invoice:
    invoice_id: str
    vendor_id: str
    invoice_number: str
    invoice_date: str
    due_date: str
    submission_date: str
    submission_hour: int
    category: str
    description: str
    amount_usd: float
    tax_amount: float
    total_amount: float
    payment_status: str
    payment_date: Optional[str]
    is_fraudulent: bool
    fraud_pattern: str
    is_duplicate: bool
    original_invoice_id: Optional[str]
    approval_limit: float
    requires_additional_approval: bool


def generate_invoice(
    invoice_num: int,
    vendor: dict,
    start_date: datetime,
    end_date: datetime,
    existing_invoices: list,
    fraud_rate: float = 0.05
) -> Invoice:

    invoice_date = fake.date_time_between(
        start_date=start_date,
        end_date=end_date
    )

    # Payment terms affect due date
    terms_days = {
        "NET_15": 15, "NET_30": 30,
        "NET_45": 45, "NET_60": 60
    }
    days = terms_days.get(vendor["payment_terms"], 30)
    due_date = invoice_date + timedelta(days=days)

    # Submission timing
    submission_date = invoice_date + timedelta(days=random.randint(0, 5))
    submission_hour = random.randint(0, 23)

    # Amount based on vendor size
    base_amount = vendor["annual_spend_usd"] / random.randint(10, 100)
    amount = round(random.uniform(base_amount * 0.5, base_amount * 1.5), 2)

    category = random.choice(INVOICE_CATEGORIES)
    tax_rate = 0.08
    tax_amount = round(amount * tax_rate, 2)
    total_amount = round(amount + tax_amount, 2)

    # Approval limit
    approval_limit = 10000.0
    requires_additional_approval = total_amount > approval_limit

    # Fraud detection
    is_fraudulent = random.random() < fraud_rate
    fraud_pattern = "none"
    is_duplicate = False
    original_invoice_id = None

    if is_fraudulent:
        pattern = random.choice(FRAUD_PATTERNS[:-1])
        fraud_pattern = pattern

        if pattern == "duplicate_invoice" and existing_invoices:
            original = random.choice(existing_invoices)
            is_duplicate = True
            original_invoice_id = original.invoice_id
            amount = original.amount_usd
            tax_amount = original.tax_amount
            total_amount = original.total_amount

        elif pattern == "round_number_amount":
            amount = round(random.choice([
                1000, 2000, 5000, 10000, 25000, 50000
            ]) * random.randint(1, 10), 2)
            tax_amount = round(amount * tax_rate, 2)
            total_amount = round(amount + tax_amount, 2)

        elif pattern == "weekend_submission":
            # Force to weekend
            while submission_date.weekday() < 5:
                submission_date += timedelta(days=1)

        elif pattern == "after_hours_submission":
            submission_hour = random.choice(
                list(range(0, 6)) + list(range(22, 24))
            )

        elif pattern == "slightly_under_approval_limit":
            amount = round(approval_limit - random.uniform(1, 500), 2)
            tax_amount = round(amount * tax_rate, 2)
            total_amount = round(amount + tax_amount, 2)
            requires_additional_approval = False

        elif pattern == "new_vendor_large_amount":
            amount = round(random.uniform(50000, 500000), 2)
            tax_amount = round(amount * tax_rate, 2)
            total_amount = round(amount + tax_amount, 2)

    # Payment status
    today = datetime.now()
    if invoice_date < today - timedelta(days=days + 30):
        payment_status = random.choices(
            PAYMENT_STATUSES,
            weights=[70, 10, 10, 5, 5]
        )[0]
    elif invoice_date < today - timedelta(days=days):
        payment_status = random.choices(
            PAYMENT_STATUSES,
            weights=[30, 30, 30, 5, 5]
        )[0]
    else:
        payment_status = random.choices(
            PAYMENT_STATUSES,
            weights=[10, 70, 5, 10, 5]
        )[0]

    payment_date = None
    if payment_status == "PAID":
        payment_date = (
            invoice_date + timedelta(days=random.randint(1, days + 10))
        ).date().isoformat()

    return Invoice(
        invoice_id=f"INV-{invoice_num:06d}",
        vendor_id=vendor["vendor_id"],
        invoice_number=f"{vendor['vendor_id']}-{fake.bothify('??##??##')}",
        invoice_date=invoice_date.date().isoformat(),
        due_date=due_date.date().isoformat(),
        submission_date=submission_date.date().isoformat(),
        submission_hour=submission_hour,
        category=category,
        description=fake.bs().title(),
        amount_usd=amount,
        tax_amount=tax_amount,
        total_amount=total_amount,
        payment_status=payment_status,
        payment_date=payment_date,
        is_fraudulent=is_fraudulent,
        fraud_pattern=fraud_pattern,
        is_duplicate=is_duplicate,
        original_invoice_id=original_invoice_id,
        approval_limit=approval_limit,
        requires_additional_approval=requires_additional_approval
    )


def generate_transactions(
    vendors: list[dict],
    n: int = 50000
) -> list[dict]:
    start_date = datetime.now() - timedelta(days=365 * 2)
    end_date = datetime.now()

    invoices = []
    invoice_num = 1

    print(f"Generating {n} transactions...")

    for i in range(n):
        vendor = random.choice(vendors)
        invoice = generate_invoice(
            invoice_num=invoice_num,
            vendor=vendor,
            start_date=start_date,
            end_date=end_date,
            existing_invoices=invoices[-100:],
            fraud_rate=0.05
        )
        invoices.append(invoice)
        invoice_num += 1

        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1}/{n} transactions...")

    return [asdict(inv) for inv in invoices]


def save_transactions_csv(
    transactions: list[dict],
    path: str = "data/raw/csv/transactions.csv"
):
    if not transactions:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=transactions[0].keys())
        writer.writeheader()
        writer.writerows(transactions)

    print(f"Saved {len(transactions)} transactions to {path}")


if __name__ == "__main__":
    # Load vendors
    with open("data/raw/csv/vendors.json") as f:
        vendors = json.load(f)

    # Only use active vendors
    active_vendors = [v for v in vendors if v["active"]]
    print(f"Using {len(active_vendors)} active vendors")

    # Generate transactions
    transactions = generate_transactions(active_vendors, n=50000)

    # Save
    save_transactions_csv(transactions)

    # Stats
    fraudulent = sum(1 for t in transactions if t["is_fraudulent"])
    duplicates = sum(1 for t in transactions if t["is_duplicate"])
    print(f"Fraud rate: {fraudulent/len(transactions)*100:.1f}%")
    print(f"Duplicate rate: {duplicates/len(transactions)*100:.1f}%")
    print(f"Sample: {transactions[0]}")