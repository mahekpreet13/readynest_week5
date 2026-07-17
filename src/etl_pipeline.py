"""
Phase 2: ETL Pipeline
=====================
Extract → Transform → Load

Reads raw e-commerce CSV, cleans it, engineers features, and writes
the cleaned dataset to data/cleaned/.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH     = Path(__file__).parent.parent / "data" / "ecommerce_sales_jharkhand.csv"
CLEANED_PATH = Path(__file__).parent.parent / "data" / "cleaned"
CLEANED_PATH.mkdir(parents=True, exist_ok=True)


def ensure_raw_data(path: Path = RAW_PATH) -> Path:
    """Check if the dataset exists."""
    if path.exists():
        return path
    raise FileNotFoundError(f"Raw dataset not found at {path}")


# ═══════════════════════════════════════════════════════════════════
# EXTRACT
# ═══════════════════════════════════════════════════════════════════

def extract(path: Path = RAW_PATH) -> pd.DataFrame:
    print("📥  EXTRACT: Reading raw data …")
    ensure_raw_data(path)
    df = pd.read_csv(path)
    print(f"    Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"    Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ═══════════════════════════════════════════════════════════════════
# TRANSFORM
# ═══════════════════════════════════════════════════════════════════

def transform(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧  TRANSFORM: Cleaning & engineering features …")
    original_shape = df.shape

    # Rename columns to standard lowercase
    df = df.rename(columns={
        "Order_ID": "order_id",
        "Order_Date": "order_date",
        "Customer_Name": "customer_name",
        "Price": "unit_price",
        "Discount_Percent": "discount_pct",
        "Payment_Mode": "payment_method",
        "Delivery_Status": "status",
        "Quantity": "quantity",
        "Revenue": "revenue",
        "City": "city",
        "District": "district",
        "State": "state",
        "Category": "category",
        "Product": "product"
    })

    # ── 1. Fix data types ───────────────────────────────────────────
    df["order_date"]    = pd.to_datetime(df["order_date"])
    df["unit_price"]    = pd.to_numeric(df["unit_price"],   errors="coerce")
    df["revenue"]       = pd.to_numeric(df["revenue"],      errors="coerce")

    # ── 2. Handle missing values ────────────────────────────────────
    # Drop rows with critical null values
    before = len(df)
    df.dropna(subset=["order_id", "order_date", "revenue"], inplace=True)
    print(f"    Dropped {before - len(df)} rows with critical nulls")

    # ── 3. Remove exact duplicates ──────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=["order_id"], inplace=True)
    print(f"    Removed {before - len(df)} duplicate order_ids")

    # ── 4. Fix negative / impossible values ─────────────────────────
    df.loc[df["unit_price"] < 0, "unit_price"] = np.nan
    df.loc[df["quantity"]   < 1, "quantity"]   = 1
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())

    # ── 5. Date-based feature engineering ──────────────────────────
    df["order_year"]    = df["order_date"].dt.year
    df["order_month"]   = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_week"]    = df["order_date"].dt.isocalendar().week.astype(int)
    df["order_dow"]     = df["order_date"].dt.dayofweek          # 0=Mon
    df["order_dow_name"]= df["order_date"].dt.day_name()
    df["is_weekend"]    = df["order_dow"].isin([5, 6]).astype(int)

    # ── 6. Revenue features ───────────────────────────────
    df["revenue_per_unit"]  = (df["revenue"] / df["quantity"]).round(2)

    # ── 7. Order status flags ───────────────────────────────────────
    df["is_cancelled"]  = (df["status"] == "Cancelled").astype(int)
    df["is_completed"]  = (df["status"] == "Delivered").astype(int)

    # ── 8. Discount effectiveness ───────────────────────────────────
    df["has_discount"] = (df["discount_pct"] > 0).astype(int)

    # ── 9. Anomaly flag (revenue outliers) ─────────────────────────
    q1 = df["revenue"].quantile(0.25)
    q3 = df["revenue"].quantile(0.75)
    iqr = q3 - q1
    df["is_anomaly"] = (
        (df["revenue"] < q1 - 3 * iqr) |
        (df["revenue"] > q3 + 3 * iqr)
    ).astype(int)

    print(f"    Shape: {original_shape} → {df.shape}")
    print(f"    Anomalies flagged: {df['is_anomaly'].sum()}")
    print(f"    New features added: order_year, order_month, order_quarter, "
          "order_week, order_dow, is_weekend, revenue_per_unit, "
          "is_cancelled, is_completed, has_discount, is_anomaly")

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════

def load(df: pd.DataFrame) -> Path:
    print("\n💾  LOAD: Saving cleaned dataset …")
    out = CLEANED_PATH / "ecommerce_cleaned.csv"
    df.to_csv(out, index=False)
    print(f"    ✅  Saved → {out}  ({len(df):,} rows × {len(df.columns)} cols)")

    # Also save monthly aggregated summary
    monthly = (
        df.groupby(["order_year", "order_month"])
          .agg(total_orders=("order_id",  "count"),
               total_revenue=("revenue",  "sum"),
               avg_order_value=("revenue", "mean"))
          .reset_index()
    )
    monthly.to_csv(CLEANED_PATH / "monthly_summary.csv", index=False)
    print(f"    ✅  Monthly summary → {CLEANED_PATH / 'monthly_summary.csv'}")
    return out


# ═══════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_pipeline() -> pd.DataFrame:
    print("=" * 60)
    print("     E-COMMERCE ETL PIPELINE")
    print("=" * 60)
    df_raw     = extract()
    df_clean   = transform(df_raw)
    load(df_clean)
    print("\n🎉  ETL Pipeline complete!")
    return df_clean


if __name__ == "__main__":
    run_pipeline()
