"""
Vendor Generator
Generates synthetic vendor master data for the Financial Document Intelligence platform.
500 vendors across multiple industries, regions, and risk profiles.
"""

import random
import json
from faker import Faker
from dataclasses import dataclass, asdict
from typing import Optional

fake = Faker()
random.seed(42)
Faker.seed(42)

INDUSTRIES = [
    "Technology", "Healthcare", "Manufacturing", "Retail",
    "Logistics", "Finance", "Construction", "Food & Beverage",
    "Energy", "Consulting"
]

REGIONS = [
    "Northeast", "Southeast", "Midwest", "Southwest",
    "West Coast", "Mountain", "Plains", "Mid-Atlantic"
]

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]

CONTRACT_TIERS = ["BRONZE", "SILVER", "GOLD", "PLATINUM"]

PAYMENT_TERMS = ["NET_15", "NET_30", "NET_45", "NET_60"]


@dataclass
class Vendor:
    vendor_id: str
    vendor_name: str
    industry: str
    region: str
    country: str
    city: str
    state: str
    zip_code: str
    contact_email: str
    contact_phone: str
    risk_score: float
    risk_level: str
    contract_tier: str
    payment_terms: str
    annual_spend_usd: float
    active: bool
    onboarded_date: str


def generate_vendor(vendor_num: int) -> Vendor:
    industry = random.choice(INDUSTRIES)
    risk_score = round(random.uniform(0.1, 9.9), 2)

    if risk_score < 3.0:
        risk_level = "LOW"
    elif risk_score < 7.0:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return Vendor(
        vendor_id=f"VND-{vendor_num:04d}",
        vendor_name=fake.company(),
        industry=industry,
        region=random.choice(REGIONS),
        country="United States",
        city=fake.city(),
        state=fake.state_abbr(),
        zip_code=fake.zipcode(),
        contact_email=fake.company_email(),
        contact_phone=fake.phone_number(),
        risk_score=risk_score,
        risk_level=risk_level,
        contract_tier=random.choice(CONTRACT_TIERS),
        payment_terms=random.choice(PAYMENT_TERMS),
        annual_spend_usd=round(random.uniform(10000, 5000000), 2),
        active=random.choices([True, False], weights=[90, 10])[0],
        onboarded_date=fake.date_between(
            start_date="-5y", end_date="-30d"
        ).isoformat()
    )


def generate_vendors(n: int = 500) -> list[dict]:
    vendors = [asdict(generate_vendor(i + 1)) for i in range(n)]
    return vendors


def save_vendors(vendors: list[dict], path: str = "data/raw/csv/vendors.json"):
    with open(path, "w") as f:
        json.dump(vendors, f, indent=2)
    print(f"Saved {len(vendors)} vendors to {path}")


if __name__ == "__main__":
    vendors = generate_vendors(500)
    save_vendors(vendors)

    # Quick stats
    risk_counts = {}
    for v in vendors:
        risk_counts[v["risk_level"]] = risk_counts.get(v["risk_level"], 0) + 1
    print(f"Risk distribution: {risk_counts}")
    print(f"Sample vendor: {vendors[0]}")