"""
Phase 4: Predictive Model
==========================
Trains a lightweight regression model to predict order revenue.
The implementation avoids optional external ML dependencies so the
project can run in a minimal Python environment.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Keep matplotlib caches inside the workspace so imports do not fail on
# machines without a writable global font/cache directory.
BASE_DIR = Path(__file__).parent.parent
MPLCONFIG_DIR = BASE_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import warnings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEANED_CSV = BASE_DIR / "data" / "cleaned" / "ecommerce_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = BASE_DIR / "reports" / "charts"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
BG_COLOR = "#0D1117"
TEXT_COLOR = "#E6EDF3"
PALETTE = ["#00C896", "#1DB954", "#0077B6", "#023E8A", "#48CAE4"]

# ── Feature preparation ───────────────────────────────────────────────────────
NUMERIC_COLS = [
    "unit_price",
    "quantity",
    "discount_pct",
    "order_month",
    "order_quarter",
    "order_dow",
    "is_weekend",
    "has_discount",
]
CAT_COLS = ["category", "city", "payment_method"]
TARGET_COL = "revenue"


def ensure_cleaned_data() -> None:
    """Run ETL if the cleaned dataset is missing."""
    if CLEANED_CSV.exists():
        return

    print("⚠️  Cleaned dataset not found — running ETL pipeline …")
    try:
        from etl_pipeline import run_pipeline
    except ImportError:
        from src.etl_pipeline import run_pipeline

    run_pipeline()


def load_data() -> pd.DataFrame:
    ensure_cleaned_data()
    return pd.read_csv(CLEANED_CSV)


def prepare_features(df: pd.DataFrame):
    """Build a one-hot encoded design matrix."""
    df_model = df[df["status"] == "Delivered"].copy()
    df_model = df_model[df_model[TARGET_COL] > 0].copy()
    df_model = df_model[NUMERIC_COLS + CAT_COLS + [TARGET_COL]].dropna()

    X = pd.get_dummies(df_model[NUMERIC_COLS + CAT_COLS], columns=CAT_COLS, drop_first=False)
    X = X.astype(float)
    y = df_model[TARGET_COL].to_numpy(dtype=float)
    feature_columns = list(X.columns)
    return X, y, feature_columns, df_model


def train_test_split_frame(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Deterministic shuffle split without sklearn."""
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    split_idx = max(1, int(len(indices) * (1 - test_size)))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]


def fit_ridge_regression(X: pd.DataFrame, y: np.ndarray, alpha: float = 5.0) -> dict:
    """Fit a stable linear model with small L2 regularization."""
    X_mat = X.to_numpy(dtype=float)
    y_vec = np.asarray(y, dtype=float)
    X_aug = np.column_stack([np.ones(len(X_mat)), X_mat])

    penalty = np.eye(X_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(X_aug.T @ X_aug + penalty) @ (X_aug.T @ y_vec)

    return {
        "intercept": float(beta[0]),
        "coef": beta[1:].astype(float),
        "feature_columns": list(X.columns),
        "alpha": alpha,
        "model_type": "ridge_linear_regression",
    }


def predict(model: dict, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    X_mat = np.asarray(X, dtype=float)
    return model["intercept"] + X_mat @ np.asarray(model["coef"], dtype=float)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def cross_val_r2(X: pd.DataFrame, y: np.ndarray, folds: int = 5, alpha: float = 5.0, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    fold_indices = np.array_split(indices, folds)

    scores = []
    for i in range(folds):
        test_idx = fold_indices[i]
        train_idx = np.concatenate([fold_indices[j] for j in range(folds) if j != i])

        fold_model = fit_ridge_regression(X.iloc[train_idx], y[train_idx], alpha=alpha)
        fold_pred = predict(fold_model, X.iloc[test_idx])
        scores.append(r2_score_np(y[test_idx], fold_pred))

    return np.asarray(scores, dtype=float)


def evaluate(model: dict, X_test: pd.DataFrame, y_test: np.ndarray):
    y_pred = predict(model, X_test)
    metrics = {
        "rmse": rmse(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "r2": r2_score_np(y_test, y_pred),
    }

    print("\n" + "=" * 55)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Algorithm : {model['model_type']}")
    print(f"  Test size : {len(y_test):,} samples")
    print(f"  RMSE      : ${metrics['rmse']:,.2f}")
    print(f"  MAE       : ${metrics['mae']:,.2f}")
    print(f"  R² Score  : {metrics['r2']:.4f}  ({metrics['r2']*100:.1f}% variance explained)")
    print("=" * 55)
    return y_pred, metrics


def plot_feature_importance(model: dict):
    importances = np.abs(np.asarray(model["coef"], dtype=float))
    total = importances.sum()
    if total:
        importances = importances / total

    fi = pd.Series(importances, index=model["feature_columns"]).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    colors = plt.cm.YlGn(np.linspace(0.3, 0.9, len(fi)))
    bars = ax.barh(fi.index, fi.values, color=colors, edgecolor="#30363D")
    ax.set_xlabel("Relative Importance", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.set_facecolor("#161B22")
    ax.set_title("Feature Importance - Revenue Prediction Model", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    for b, val in zip(bars, fi.values):
        ax.text(val + 0.001, b.get_y() + b.get_height() / 2, f"{val:.3f}", va="center", color=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    path = REPORTS_DIR / "09_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"    📊  Feature importance chart saved -> {path.name}")


def plot_predictions(y_test: np.ndarray, y_pred: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG_COLOR)

    ax1.scatter(y_test, y_pred, alpha=0.35, color=PALETTE[0], s=18, edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax1.set_xlabel("Actual Revenue ($)", color=TEXT_COLOR)
    ax1.set_ylabel("Predicted Revenue ($)", color=TEXT_COLOR)
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.set_facecolor("#161B22")
    ax1.set_title("Actual vs Predicted Revenue", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, facecolor="#161B22", labelcolor=TEXT_COLOR)
    for s in ax1.spines.values():
        s.set_edgecolor("#30363D")

    residuals = y_test - y_pred
    ax2.hist(residuals, bins=50, color=PALETTE[2], edgecolor="#30363D", alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--", lw=1.5)
    ax2.set_xlabel("Residual (Actual - Predicted)", color=TEXT_COLOR)
    ax2.set_ylabel("Count", color=TEXT_COLOR)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.set_facecolor("#161B22")
    ax2.set_title("Residual Distribution", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    for s in ax2.spines.values():
        s.set_edgecolor("#30363D")

    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    path = REPORTS_DIR / "10_actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"    📊  Predictions chart saved -> {path.name}")


def run_model() -> dict:
    print("=" * 55)
    print("  PREDICTIVE MODEL PIPELINE")
    print("=" * 55)

    print("\n📥  Loading cleaned data …")
    df = load_data()
    print(f"    {len(df):,} rows loaded")

    print("\n🔧  Preparing features …")
    X, y, feature_columns, df_model = prepare_features(df)
    print(f"    Training set: {len(X):,} samples × {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split_frame(X, y, test_size=0.2, random_state=42)

    print("\n🔄  Running 5-fold cross-validation …")
    cv_scores = cross_val_r2(X_train, y_train, folds=5, alpha=5.0, random_state=42)
    print(f"    CV R² scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"    Mean CV R²  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n🚀  Training final model …")
    model = fit_ridge_regression(X_train, y_train, alpha=5.0)

    y_pred, metrics = evaluate(model, X_test, y_test)

    print("\n  Generating model charts …")
    plot_feature_importance(model)
    plot_predictions(y_test, y_pred)

    model_path = MODEL_DIR / "revenue_model.pkl"
    artifact = {
        "model": model,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CAT_COLS,
        "target_col": TARGET_COL,
        "sample_rows": len(df_model),
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\n💾  Model saved -> {model_path}")
    print("\n🎉  Model training complete!")
    return metrics


if __name__ == "__main__":
    run_model()
