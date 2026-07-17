"""
Phase 1: Data Collection
========================
Generates a realistic synthetic e-commerce sales dataset and saves it to data/raw/.
Simulates data that could be collected from an e-commerce API or web scraping.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime, timedelta

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────
NUM_RECORDS = 5000
START_DATE  = datetime(2023, 1, 1)
END_DATE    = datetime(2024, 12, 31)
OUTPUT_DIR  = Path(__file__).parent.parent / "data" / "raw"

# ── Lookup tables ─────────────────────────────────────────────────────────────
CATEGORIES = {
    "Electronics":   {"products": ["Laptop", "Smartphone", "Headphones", "Tablet", "Camera", "Smartwatch"],
                      "price_range": (199, 1999), "margin": (0.10, 0.25)},
    "Clothing":      {"products": ["T-Shirt", "Jeans", "Jacket", "Dress", "Sneakers", "Hoodie"],
                      "price_range": (15,  250), "margin": (0.35, 0.65)},
    "Home & Garden": {"products": ["Sofa", "Lamp", "Bed Frame", "Cookware Set", "Curtains", "Rug"],
                      "price_range": (25,  899), "margin": (0.20, 0.45)},
    "Books":         {"products": ["Fiction Novel", "Textbook", "Self-Help", "Cookbook", "Biography", "Comics"],
                      "price_range": (5,   80),  "margin": (0.25, 0.50)},
    "Sports":        {"products": ["Yoga Mat", "Dumbbell Set", "Running Shoes", "Bicycle", "Tent", "Protein Powder"],
                      "price_range": (20,  599), "margin": (0.18, 0.40)},
    "Beauty":        {"products": ["Perfume", "Skincare Kit", "Lipstick", "Hair Dryer", "Face Wash", "Moisturizer"],
                      "price_range": (10,  299), "margin": (0.40, 0.70)},
    "Toys":          {"products": ["LEGO Set", "Action Figure", "Board Game", "Remote Car", "Doll", "Puzzle"],
                      "price_range": (10,  199), "margin": (0.28, 0.55)},
    "Groceries":     {"products": ["Coffee Beans", "Organic Tea", "Olive Oil", "Pasta", "Cereal", "Juice"],
                      "price_range": (3,   60),  "margin": (0.12, 0.30)},
}

REGIONS      = ["North", "South", "East", "West", "Central"]
CHANNELS     = ["Online", "Mobile App", "In-Store", "Marketplace"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "UPI", "Net Banking", "Cash on Delivery"]
CUSTOMER_TIERS  = ["Bronze", "Silver", "Gold", "Platinum"]

# ── Helper functions ──────────────────────────────────────────────────────────

def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_customer_id() -> str:
    return f"CUST-{random.randint(10000, 99999)}"


def generate_order_id(idx: int) -> str:
    return f"ORD-{100000 + idx}"


# ── Main generation ───────────────────────────────────────────────────────────

def generate_dataset(n: int = NUM_RECORDS) -> pd.DataFrame:
    records = []

    # Weighted category distribution (Electronics & Clothing sell more)
    cat_weights = [3, 3, 2, 1, 2, 2, 1, 2]
    categories  = list(CATEGORIES.keys())

    for i in range(n):
        category    = random.choices(categories, weights=cat_weights, k=1)[0]
        cat_info    = CATEGORIES[category]
        product     = random.choice(cat_info["products"])
        unit_price  = round(random.uniform(*cat_info["price_range"]), 2)
        quantity    = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 12, 8, 5], k=1)[0]
        margin_pct  = random.uniform(*cat_info["margin"])

        order_date   = random_date(START_DATE, END_DATE)
        delivery_days = random.randint(1, 14)
        delivery_date = order_date + timedelta(days=delivery_days)

        discount_pct = random.choices([0, 5, 10, 15, 20, 25, 30],
                                      weights=[40, 15, 15, 10, 10, 5, 5], k=1)[0]
        discount_amt = round(unit_price * quantity * discount_pct / 100, 2)
        revenue      = round(unit_price * quantity - discount_amt, 2)
        cost         = round(revenue * (1 - margin_pct), 2)
        profit       = round(revenue - cost, 2)

        # Inject ~3% anomalies (returns / cancellations)
        status = random.choices(
            ["Delivered", "Shipped", "Processing", "Cancelled", "Returned"],
            weights=[65, 15, 8, 7, 5], k=1
        )[0]
        if status in ("Cancelled", "Returned"):
            revenue = 0.0
            profit  = -cost * 0.05   # restocking fee loss

        records.append({
            "order_id":        generate_order_id(i),
            "customer_id":     generate_customer_id(),
            "customer_tier":   random.choice(CUSTOMER_TIERS),
            "order_date":      order_date.strftime("%Y-%m-%d"),
            "delivery_date":   delivery_date.strftime("%Y-%m-%d"),
            "category":        category,
            "product":         product,
            "unit_price":      unit_price,
            "quantity":        quantity,
            "discount_pct":    discount_pct,
            "discount_amt":    discount_amt,
            "revenue":         revenue,
            "cost":            cost,
            "profit":          profit,
            "region":          random.choice(REGIONS),
            "channel":         random.choice(CHANNELS),
            "payment_method":  random.choice(PAYMENT_METHODS),
            "delivery_days":   delivery_days,
            "status":          status,
            "rating":          round(random.uniform(1.0, 5.0), 1) if status == "Delivered" else None,
        })

    return pd.DataFrame(records)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("🔄  Generating e-commerce sales dataset …")
    df = generate_dataset(NUM_RECORDS)

    out_path = OUTPUT_DIR / "ecommerce_sales.csv"
    df.to_csv(out_path, index=False)

    print(f"✅  Raw dataset saved → {out_path}")
    print(f"    Rows     : {len(df):,}")
    print(f"    Columns  : {len(df.columns)}")
    print(f"    Date span: {df['order_date'].min()} → {df['order_date'].max()}")
    print(f"\nSample:\n{df.head(3).to_string(index=False)}")
    return df


if __name__ == "__main__":
    main()
