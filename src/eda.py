"""Clean the Telco Customer Churn dataset and report basic data quality."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = ROOT / "data" / "cleaned_telco_churn.csv"


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}. See data/README.md for download instructions."
        )

    df = pd.read_csv(INPUT_PATH)
    print("Initial dataset shape:", df.shape)
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    df = df.drop_duplicates().copy()

    # TotalCharges is stored as text in the original dataset.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Customers with zero tenure have no accumulated charges.
    mask = df["tenure"].eq(0) & df["TotalCharges"].isna()
    cleaned_zero_tenure = int(mask.sum())
    df.loc[mask, "TotalCharges"] = 0

    print("\nMissing values after cleaning:")
    print(df["TotalCharges"].isnull().sum())
    print("\nZero-tenure records cleaned:")
    print(cleaned_zero_tenure)
    print("\nUpdated data types:")
    print(df["TotalCharges"].dtype)
    print("\nFinal shape:", df.shape)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("\nCleaned dataset saved successfully!")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
