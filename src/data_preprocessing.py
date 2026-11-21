"""Reusable loading and inspection helpers for telecom churn data."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

TARGET = "Churn"


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV and normalize column labels."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. See data/README.md for setup instructions."
        )
    frame = pd.read_csv(file_path)
    frame.columns = [str(column).strip().replace(" ", "_") for column in frame.columns]
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET}' in the dataset.")
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame = frame.drop_duplicates().copy()
    frame[TARGET] = frame[TARGET].astype(str).str.strip()
    frame = frame[frame[TARGET].isin(["Yes", "No"])].copy()
    if frame.empty:
        raise ValueError("No valid rows remain after basic data cleaning.")
    return frame


def inspect_dataset(frame: pd.DataFrame) -> dict:
    """Return a JSON-friendly data quality and numeric outlier summary."""
    numeric = frame.select_dtypes(include="number")
    outliers: dict[str, dict[str, float | int]] = {}
    for column in numeric.columns:
        q1 = numeric[column].quantile(0.25)
        q3 = numeric[column].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((numeric[column] < lower) | (numeric[column] > upper)).sum())
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
