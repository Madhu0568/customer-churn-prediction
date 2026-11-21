"""Train and evaluate customer churn classification models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data_preprocessing import TARGET, load_dataset, inspect_dataset

ROOT = Path(__file__).resolve().parents[1]


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap numeric outliers at training-fold IQR bounds."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1 = np.percentile(values, 25, axis=0)
        q3 = np.percentile(values, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - 1.5 * iqr
        self.upper_bounds_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_capping", IQRCapper()),
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


def save_eda_charts(frame: pd.DataFrame, reports_dir: Path) -> None:
    """Save small EDA charts to the reports folder."""
    counts = frame[TARGET].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    fig.savefig(reports_dir / "churn_distribution.png", dpi=150)
    plt.close(fig)

    if "tenure" in frame.columns and frame["tenure"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 4))
        frame.boxplot(column="tenure", by=TARGET, ax=ax)
        fig.suptitle("")
        ax.set_title("Customer Tenure by Churn Status")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Tenure (months)")
        fig.tight_layout()
        fig.savefig(reports_dir / "tenure_by_churn.png", dpi=150)
        plt.close(fig)


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
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "data_quality.json").write_text(
        json.dumps(inspection, indent=2), encoding="utf-8"
    )
    save_eda_charts(frame, reports_dir)

    # customerID is an identifier, not a predictive measurement.
    X = frame.drop(columns=[TARGET, "customerID"], errors="ignore")
    y = frame[TARGET].map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, class_weight="balanced",
            random_state=args.random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=args.random_state, n_jobs=-1
        ),
    }

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    for name, estimator in estimators.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", estimator),
        ])
        # Select the model using training-only cross-validation, then evaluate
        # that fitted pipeline on the held-out test data.
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
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
        print(f"  cross-validation ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} "
              f"+/- {metrics['cv_roc_auc_std']:.4f}")
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            print(f"  {metric}: {metrics[metric]:.4f}")
        print("  confusion matrix [ [TN, FP], [FN, TP] ]:")
        print(metrics["confusion_matrix"])

    best_name = max(results, key=lambda key: results[key]["cv_roc_auc_mean"])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], ROOT / "models" / "churn_model.joblib")
    payload = {
        "selected_model": best_name,
        "selection_metric": "mean 5-fold training cross-validation ROC-AUC",
        "models": results,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipeline in fitted.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC Curves — Customer Churn Models")
    fig.tight_layout()
    fig.savefig(reports_dir / "roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"\nSelected model: {best_name}")
    print("Saved model: models/churn_model.joblib")
    print("Saved metrics: reports/metrics.json")


if __name__ == "__main__":
    main()
