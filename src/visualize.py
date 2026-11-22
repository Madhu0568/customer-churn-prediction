"""Create exploratory visualizations for customer churn."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_telco_churn.csv"
REPORTS_DIR = ROOT / "reports"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. Run python -m src.eda first."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    churn_counts.plot(kind="bar", ax=ax)
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "churn_distribution.png", dpi=150)
    plt.close(fig)
    print("Churn distribution chart saved!")

    contract_churn = pd.crosstab(
        df["Contract"], df["Churn"], normalize="index"
    ).mul(100)
    print("\nChurn rate by contract type:")
    print(contract_churn.round(2))

    ax = contract_churn.reindex(columns=["No", "Yes"])["Yes"].plot(
        kind="bar",
        figsize=(8, 5),
        title="Churn Rate by Contract Type",
        ylabel="Churn Rate (%)",
        xlabel="Contract Type",
    )
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "churn_by_contract.png", dpi=150)
    plt.close(fig)
    print("\nContract churn chart saved!")


if __name__ == "__main__":
    main()
