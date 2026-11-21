"""Reusable loading, feature engineering, and inspection helpers."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

TARGET = "Churn"
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load, clean, and engineer deterministic customer-level features."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. See data/README.md for setup instructions."
        )
    frame = pd.read_csv(file_path)
    frame.columns = [str(column).strip().replace(" ", "_") for column in frame.columns]
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET}' in the dataset.")

    # Convert fields that should be numeric; blank TotalCharges values become NaN.
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.drop_duplicates().copy()
    frame[TARGET] = frame[TARGET].astype(str).str.strip()
    frame = frame[frame[TARGET].isin(["Yes", "No"])].copy()
    if frame.empty:
        raise ValueError("No valid rows remain after basic data cleaning.")

    # Feature engineering: tenure bands, number of subscribed services, and
    # average billed amount per active month. Division by zero is handled safely.
    if "tenure" in frame.columns:
        frame["tenure_group"] = pd.cut(
            frame["tenure"],
            bins=[-1, 12, 24, 48, 60, float("inf")],
            labels=["0-12", "13-24", "25-48", "49-60", "60+"],
        ).astype("object")
    service_columns = [column for column in SERVICE_COLUMNS if column in frame.columns]
    if service_columns:
        frame["service_count"] = (
            frame[service_columns].astype(str).apply(lambda col: col.str.strip().eq("Yes")).sum(axis=1)
        )
    if {"TotalCharges", "tenure"}.issubset(frame.columns):
        safe_tenure = frame["tenure"].where(frame["tenure"] > 0)
        frame["average_charge_per_month"] = frame["TotalCharges"] / safe_tenure

    return frame


def inspect_dataset(frame: pd.DataFrame) -> dict:
    """Return a JSON-friendly data quality and numeric IQR outlier summary."""
    numeric = frame.select_dtypes(include="number")
    outliers: dict[str, dict[str, float | int]] = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        outliers[column] = {
            "lower_iqr_bound": float(lower),
            "upper_iqr_bound": float(upper),
            "outlier_count": count,
        }
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing_values": {k: int(v) for k, v in frame.isna().sum().items()},
        "duplicate_rows_after_cleaning": int(frame.duplicated().sum()),
        "target_distribution": {k: int(v) for k, v in frame[TARGET].value_counts().items()},
        "iqr_outlier_summary": outliers,
    }
