"""Generate churn predictions from a saved model and customer CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from .data_preprocessing import engineer_features

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict customer churn from a CSV file.")
    parser.add_argument("--input", required=True, help="CSV containing customer feature columns.")
    parser.add_argument("--output", default="reports/predictions.csv")
    parser.add_argument("--model", default="models/churn_model.joblib")
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError("Trained model not found. Run python -m src.train first.")

    customers = pd.read_csv(input_path)
    customers = engineer_features(customers)
    customers = customers.drop(columns=["Churn", "customerID"], errors="ignore")
    model = joblib.load(model_path)
    labels = model.predict(customers)
    probabilities = model.predict_proba(customers)[:, 1]
    output = pd.DataFrame({
        "churn_prediction": ["Yes" if value == 1 else "No" for value in labels],
        "churn_probability": probabilities,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output)} predictions to {output_path}")


if __name__ == "__main__":
    main()
