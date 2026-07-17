"""
Phase 3: Exploratory Data Analysis (EDA)
=========================================
Generates charts and prints statistical summaries to understand:
  - Revenue trends over time
  - Category performance
  - City segmentation
  - Correlations
  - Anomalies
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MPLCONFIG_DIR = BASE_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEANED_PATH = BASE_DIR / "data" / "cleaned" / "ecommerce_cleaned.csv"
REPORTS_DIR  = BASE_DIR / "reports" / "charts"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
PALETTE  = ["#00C896", "#1DB954", "#0077B6", "#023E8A", "#48CAE4", "#90E0EF", "#ADE8F4", "#CAF0F8"]
BG_COLOR = "#0D1117"
TEXT_COLOR= "#E6EDF3"

def _style_ax(ax, title: str = ""):
    ax.set_facecolor("#161B22")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")


def _save(fig, name: str):
    path = REPORTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"    📊  Saved → {path.name}")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_PATH, parse_dates=["order_date"])
    return df


# ── 1. Revenue Trend (Monthly) ────────────────────────────────────────────────
def plot_revenue_trend(df: pd.DataFrame):
    monthly = (df[df["status"] == "Delivered"]
               .groupby(["order_year", "order_month"])["revenue"]
               .sum()
               .reset_index())
    monthly["period"] = pd.to_datetime(
        monthly["order_year"].astype(str) + "-" + monthly["order_month"].astype(str).str.zfill(2)
    )
    monthly.sort_values("period", inplace=True)

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=BG_COLOR)
    ax.fill_between(monthly["period"], monthly["revenue"], alpha=0.25, color=PALETTE[0])
    ax.plot(monthly["period"], monthly["revenue"], color=PALETTE[0], lw=2.5, marker="o", ms=4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_xlabel("Month", color=TEXT_COLOR)
    ax.set_ylabel("Revenue", color=TEXT_COLOR)
    _style_ax(ax, "Monthly Revenue Trend")
    fig.tight_layout()
    _save(fig, "01_revenue_trend.png")


# ── 2. Revenue by Category ────────────────────────────────────────────────────
def plot_category_revenue(df: pd.DataFrame):
    cat_rev = (df[df["status"] == "Delivered"]
               .groupby("category")["revenue"]
               .sum()
               .sort_values(ascending=True))

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG_COLOR)
    bars = ax.barh(cat_rev.index, cat_rev.values, color=PALETTE[:len(cat_rev)], edgecolor="#30363D")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 500, bar.get_y() + bar.get_height() / 2,
                f"${w/1000:.1f}K", va="center", color=TEXT_COLOR, fontsize=8)
    _style_ax(ax, "Revenue by Product Category")
    fig.tight_layout()
    _save(fig, "02_category_revenue.png")


# ── 3. City Distribution ─────────────────────────────────────────────
def plot_city_distribution(df: pd.DataFrame):
    city_data = df["city"].value_counts().head(5)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(city_data))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_COLOR)

    # Pie
    wedges, texts, autotexts = ax1.pie(
        city_data, labels=city_data.index, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"color": TEXT_COLOR, "fontsize": 10}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_facecolor(BG_COLOR)
    ax1.set_title("Top Cities by Order Count", color=TEXT_COLOR, fontsize=12, fontweight="bold")

    # Avg revenue per city
    city_rev = df[df["status"] == "Delivered"].groupby("city")["revenue"].mean().sort_values().tail(5)
    ax2.barh(city_rev.index, city_rev.values, color=colors, edgecolor="#30363D")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
    _style_ax(ax2, "Avg Order Value by Top City")
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    _save(fig, "03_city_distribution.png")


# ── 4. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame):
    numeric_cols = ["unit_price", "quantity", "discount_pct", "revenue", "is_weekend"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                fmt=".2f", linewidths=0.5, linecolor="#30363D",
                ax=ax, annot_kws={"size": 8, "color": TEXT_COLOR},
                cbar_kws={"shrink": 0.8})
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    _style_ax(ax, "Feature Correlation Heatmap")
    ax.figure.axes[-1].tick_params(colors=TEXT_COLOR)
    fig.tight_layout()
    _save(fig, "04_correlation_heatmap.png")


# ── 5. Payment Method Performance ─────────────────────────────────────────────
def plot_payment_methods(df: pd.DataFrame):
    pm = (df[df["status"] == "Delivered"]
          .groupby("payment_method")
          .agg(total_revenue=("revenue", "sum"),
               total_orders=("order_id", "count"),
               avg_order=("revenue", "mean"))
          .sort_values("total_revenue", ascending=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=BG_COLOR)
    metrics = [("total_revenue", "Total Revenue", "$"),
               ("total_orders",  "Total Orders",  ""),
               ("avg_order",     "Avg Order Value","$")]

    for ax, (col, label, prefix) in zip(axes, metrics):
        bars = ax.bar(pm.index, pm[col], color=PALETTE[:len(pm)], edgecolor="#30363D")
        ax.set_xlabel("Payment Method", color=TEXT_COLOR)
        if prefix == "$":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K" if x >= 1000 else f"${x:.0f}"))
        for b in bars:
            h = b.get_height()
            val = f"{prefix}{h/1000:.1f}K" if (prefix == "$" and h >= 1000) else f"{prefix}{h:.0f}"
            ax.text(b.get_x() + b.get_width() / 2, h * 1.02, val,
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)
        _style_ax(ax, label)
        ax.tick_params(axis="x", rotation=15)

    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    _save(fig, "05_payment_method_performance.png")


# ── 6. Discount Impact ────────────────────────────────────────────────────────
def plot_discount_impact(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG_COLOR)

    # Scatter: discount_pct vs revenue
    sample = df[df["status"] == "Delivered"].sample(min(1000, len(df)), random_state=42)
    axes[0].scatter(sample["discount_pct"], sample["revenue"],
                    alpha=0.45, color=PALETTE[2], s=20, edgecolors="none")
    if sample["discount_pct"].nunique() > 1:
        m, b = np.polyfit(sample["discount_pct"], sample["revenue"], 1)
        xline = np.linspace(0, sample["discount_pct"].max(), 200)
        axes[0].plot(xline, m * xline + b, color=PALETTE[0], lw=2, linestyle="--")
    axes[0].set_xlabel("Discount %", color=TEXT_COLOR)
    axes[0].set_ylabel("Revenue ($)", color=TEXT_COLOR)
    _style_ax(axes[0], "Discount % vs Revenue")

    # Avg revenue by discount bracket
    df2 = df[df["status"] == "Delivered"].copy()
    df2["disc_bracket"] = pd.cut(df2["discount_pct"], bins=[-1, 0, 10, 20, 30, 100],
                                  labels=["0%", "1-10%", "11-20%", "21-30%", ">30%"])
    dm = df2.groupby("disc_bracket", observed=True)["revenue"].mean()
    axes[1].bar(dm.index, dm.values, color=PALETTE[:len(dm)], edgecolor="#30363D")
    axes[1].set_xlabel("Discount Bracket", color=TEXT_COLOR)
    axes[1].set_ylabel("Avg Revenue", color=TEXT_COLOR)
    _style_ax(axes[1], "Avg Revenue by Discount Bracket")

    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    _save(fig, "06_discount_impact.png")


# ── 7. Weekday vs Weekend Sales ───────────────────────────────────────────────
def plot_weekday_pattern(df: pd.DataFrame):
    dow = (df[df["status"] == "Delivered"]
           .groupby("order_dow_name")["revenue"]
           .sum()
           .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]))

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=BG_COLOR)
    colors_dow = [PALETTE[5] if d in ("Saturday", "Sunday") else PALETTE[0] for d in dow.index]
    bars = ax.bar(dow.index, dow.values, color=colors_dow, edgecolor="#30363D")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                f"${b.get_height()/1000:.1f}K", ha="center", va="bottom",
                fontsize=8, color=TEXT_COLOR)
    _style_ax(ax, "Revenue by Day of Week  (🟢 Weekend highlighted)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save(fig, "07_weekday_pattern.png")


# ── 8. Status Distribution ────────────────────────────────────────────────────
def plot_order_status(df: pd.DataFrame):
    status_cnt = df["status"].value_counts()
    colors_s   = [PALETTE[i % len(PALETTE)] for i in range(len(status_cnt))]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG_COLOR)
    wedges, texts, autotexts = ax.pie(
        status_cnt, labels=status_cnt.index, autopct="%1.1f%%",
        colors=colors_s, startangle=90,
        textprops={"color": TEXT_COLOR, "fontsize": 10},
        wedgeprops={"edgecolor": "#0D1117", "linewidth": 2}
    )
    ax.set_title("Order Status Distribution", color=TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    _save(fig, "08_order_status.png")


# ── Statistical Summary ───────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    delivered = df[df["status"] == "Delivered"]
    print("\n" + "=" * 60)
    print("  EDA STATISTICAL SUMMARY")
    print("=" * 60)
    print(f"  Total orders         : {len(df):,}")
    print(f"  Delivered orders     : {len(delivered):,} ({len(delivered)/len(df)*100:.1f}%)")
    print(f"  Total revenue        : ${delivered['revenue'].sum():,.2f}")
    print(f"  Avg order value      : ${delivered['revenue'].mean():.2f}")
    print(f"  Anomalies flagged    : {df['is_anomaly'].sum()}")
    print(f"\n  Top 3 categories (revenue):")
    top3 = (delivered.groupby("category")["revenue"]
                     .sum().sort_values(ascending=False).head(3))
    for cat, rev in top3.items():
        print(f"    • {cat:<20} ${rev:>12,.0f}")
    print(f"\n  Top payment method : {delivered.groupby('payment_method')['revenue'].sum().idxmax()}")
    print(f"  Best city          : {delivered.groupby('city')['revenue'].sum().idxmax()}")
    print(f"  Cancel rate        : {df['is_cancelled'].mean()*100:.1f}%")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────
def run_eda() -> pd.DataFrame:
    print("🔍  Running EDA …")
    df = load_data()

    print_summary(df)

    print("\n  Generating charts …")
    plot_revenue_trend(df)
    plot_category_revenue(df)
    plot_city_distribution(df)
    plot_correlation(df)
    plot_payment_methods(df)
    plot_discount_impact(df)
    plot_weekday_pattern(df)
    plot_order_status(df)

    print(f"\n✅  All charts saved to {REPORTS_DIR}")
    return df


if __name__ == "__main__":
    run_eda()
