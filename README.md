# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to leave a service. The workflow covers data inspection, cleaning, feature engineering, exploratory analysis, model training, evaluation, and reusable model artifacts.

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


## Live prediction app

The interactive app is built with Streamlit. GitHub Pages cannot host a Python/Streamlit application directly, so deploy this repository to **Streamlit Community Cloud**.

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Madhu0568/customer-churn-prediction&branch=main&mainModule=app.py)

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
2. Select this repository, branch `main`, and app file `app.py`.
3. Click **Deploy**. Streamlit will provide a public app URL.
4. Once deployed, add that URL to the **Live app** link at the top of this README so visitors can open the predictor directly.

> The app needs a trained model at `models/churn_model.joblib`. The model file is excluded from Git by `.gitignore`; deployment setup will be completed after the dataset is added and the model-generation flow is configured.

## Dataset

This project uses the IBM Telco Customer Churn dataset. Download the CSV (commonly named `WA_Fn-UseC_-Telco-Customer-Churn.csv`) and place it in the `data/` directory. The data file is intentionally not committed to this repository; see [data/README.md](data/README.md) for details.

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

The training workflow:
1. Loads the CSV, normalizes column names, converts numeric fields, and removes duplicate rows.
2. Handles missing numeric and categorical values using training-fitted imputers.
3. Engineers tenure groups, number of subscribed services, and average charge per active month where source columns are available.
4. Inspects numeric outliers using the IQR method and saves a data-quality report.
5. Caps numeric outliers at IQR bounds inside the preprocessing pipeline; bounds are learned from each training fold only.
6. Scales numeric features and one-hot encodes categorical features.
7. Compares Logistic Regression, Decision Tree, and Random Forest using stratified 5-fold cross-validation.
8. Evaluates models on a held-out test set with accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
9. Saves EDA charts, model metrics, and the best cross-validation model.

The same feature engineering function is reused during prediction to keep training and inference columns consistent. Preprocessing is fitted on training data only to reduce data leakage. Model selection is based on cross-validation rather than the held-out test score.

## Predict with the saved model

```bash
python -m src.predict --input data/new_customers.csv --output reports/predictions.csv
```

The input CSV should contain the same original customer feature columns as the training dataset (the `Churn` target is not required). The `customerID` column is optional and is ignored.

## Optional Streamlit app

After training the model:

```bash
streamlit run app.py
```

Upload a CSV of customer records to view churn predictions and probabilities, then download the results.

## Generated outputs

- `reports/data_quality.json` — missing values, target distribution, and IQR outlier summary.
- `reports/churn_distribution.png` and `reports/tenure_by_churn.png` — exploratory data analysis charts.
- `reports/metrics.json` and `reports/roc_curves.png` — model evaluation.
- `models/churn_model.joblib` — fitted preprocessing and prediction pipeline.

## Notes

- The target is `Churn` (Yes/No).
- Actual performance depends on the dataset and should be read from the generated metrics; no accuracy figure is claimed in advance.
- This is an educational project. Predictions should not be the sole basis for decisions affecting customers.
