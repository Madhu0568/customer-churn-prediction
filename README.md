# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to leave a service. The workflow covers data inspection, cleaning, exploratory analysis, feature engineering, model training, evaluation, and reusable model artifacts.

## Project structure

```text
customer-churn-prediction/
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

This project uses the IBM Telco Customer Churn dataset. Download the CSV (commonly named `WA_Fn-UseC_-Telco-Customer-Churn.csv`) and place it in the `data/` directory. The dataset is not committed to this repository.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train and evaluate

```bash
python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The training pipeline:
1. Normalizes column names and converts `TotalCharges` to numeric.
2. Removes duplicate records and handles missing values.
3. Separates features from the churn target using a stratified train/test split.
4. Imputes numeric and categorical features inside a scikit-learn pipeline.
5. Scales numeric features and one-hot encodes categorical features.
6. Compares Logistic Regression, Decision Tree, and Random Forest classifiers.
7. Reports accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
8. Saves the best fitted pipeline to `models/churn_model.joblib` and metrics to `reports/metrics.json`.

Preprocessing is fitted on training data only to avoid data leakage. Outlier summaries are generated during data inspection; extreme values are not blindly removed because legitimate high-value customers can be important signals.

## Predict with the saved model

```bash
python -m src.predict --input data/new_customers.csv --output reports/predictions.csv
```

The input CSV should contain the same customer feature columns as the training dataset (the `Churn` target is not required).

## Optional Streamlit app

After training the model:

```bash
streamlit run app.py
```

The app lets you upload a CSV of customer records and download churn predictions.

## Notes

- The target is `Churn` (Yes/No).
- Results depend on the dataset version and train/test split. No performance figure is claimed in advance; use the generated metrics.
- Do not use model predictions as the sole basis for decisions affecting customers.
