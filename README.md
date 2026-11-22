# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to leave a service. The workflow covers data inspection, cleaning, feature engineering, exploratory analysis, model training, evaluation, and reusable model artifacts.

## Live application

**Try the deployed app:** [Customer Churn Insights ML](https://customer-churn-insights-ml.streamlit.app/)

Upload a telecom customer CSV to view predicted churn labels, churn probabilities, a prediction summary, and download the results.

## Application preview

The deployed Streamlit application provides:
- CSV upload for telecom customer records
- A prediction summary showing the number of customers processed, predicted churn count, and average churn probability
- A customer-level results table with actual churn (when included in the input), predicted churn, and churn probability
- A downloadable predictions CSV

### Verified deployment example

The IBM Telco sample dataset was processed successfully in the deployed app:
- **Customers processed:** 7,043
- **Predicted churn:** 2,885
- **Average churn probability:** 41.4%

These are example outputs from one run on the sample dataset, not model accuracy metrics.

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
│   ├── custom_transformers.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── predict.py
│   ├── train.py
│   └── visualize.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Streamlit application

Run the application locally:

```bash
streamlit run app.py
```

If the trained model is not present, the application downloads the public IBM Telco sample dataset and trains the model automatically on first startup. This allows a fresh deployment to start without committing the dataset or a serialized model file to the repository.

## Dataset

This project uses the IBM Telco Customer Churn dataset. Download the CSV (commonly named `WA_Fn-UseC_-Telco-Customer-Churn.csv`) and place it in the `data/` directory. The data file is intentionally not committed to this repository; see [data/README.md](data/README.md) for details.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.\\.venv\\Scripts\\Activate.ps1`
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

## Generated outputs

- `reports/data_quality.json` — missing values, target distribution, and IQR outlier summary.
- `reports/churn_distribution.png` and `reports/tenure_by_churn.png` — exploratory data analysis charts.
- `reports/metrics.json` and `reports/roc_curves.png` — model evaluation.
- `models/churn_model.joblib` — fitted preprocessing and prediction pipeline.

## Run the app from VS Code terminal

Open the project folder in VS Code, open **Terminal → New Terminal**, and run:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

After Streamlit starts, the terminal displays the local app address:

**Local URL:** http://localhost:8501

Press `Ctrl + C` in the terminal to stop the app.

**Deployed app:** [Customer Churn Insights ML](https://customer-churn-insights-ml.streamlit.app/)

## Notes

- The target is `Churn` (Yes/No).
- Actual performance depends on the dataset and should be read from the generated metrics; no accuracy figure is claimed in advance.
- This is an educational project. Predictions should not be the sole basis for decisions affecting customers.
