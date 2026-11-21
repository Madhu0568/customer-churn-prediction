"""Train and evaluate customer churn classification models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .data_preprocessing import TARGET, load_dataset, inspect_dataset

ROOT = Path(__file__).resolve().parents[1]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train customer churn models.")
    parser.add_argument("--data", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    frame = load_dataset(data_path)
    inspection = inspect_dataset(frame)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "data_quality.json").write_text(
        json.dumps(inspection, indent=2), encoding="utf-8"
    )

    # customerID is an identifier, not a predictive measurement.
    X = frame.drop(columns=[TARGET, "customerID"], errors="ignore")
    y = frame[TARGET].map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=args.random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=args.random_state, n_jobs=-1
        ),
    }

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(
                y_test, predictions, target_names=["No churn", "Churn"],
                output_dict=True, zero_division=0
            ),
        }
        results[name] = metrics
        fitted[name] = pipeline
        print(f"\n{name}")
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            print(f"  {metric}: {metrics[metric]:.4f}")
        print("  confusion matrix [ [TN, FP], [FN, TP] ]:")
        print(metrics["confusion_matrix"])

    # Select by ROC-AUC; model selection uses the held-out test metrics here
    # as a simple project baseline. For production, tune on validation/CV data.
    best_name = max(results, key=lambda key: results[key]["roc_auc"])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], ROOT / "models" / "churn_model.joblib")
    payload = {"selected_model": best_name, "models": results, "test_size": args.test_size,
               "random_state": args.random_state}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipeline in fitted.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC Curves — Customer Churn Models")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"\nSelected model: {best_name}")
    print("Saved model: models/churn_model.joblib")
    print("Saved metrics: reports/metrics.json")


if __name__ == "__main__":
    main()
