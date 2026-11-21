"""Optional Streamlit interface for batch churn prediction."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
st.title("Customer Churn Prediction")
st.write("Upload customer records to estimate churn probabilities with the trained model.")

if not MODEL_PATH.is_file():
    st.warning("No trained model found. First run: python -m src.train")
    st.stop()

uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
if uploaded is not None:
    data = pd.read_csv(uploaded)
    data.columns = [str(c).strip().replace(" ", "_") for c in data.columns]
    features = data.drop(columns=["Churn", "customerID"], errors="ignore")
    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(features)[:, 1]
    labels = model.predict(features)
    result = data.copy()
    result["churn_prediction"] = ["Yes" if value == 1 else "No" for value in labels]
    result["churn_probability"] = probabilities
    st.subheader("Prediction summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", len(result))
    col2.metric("Predicted churn", int((labels == 1).sum()))
    col3.metric("Average churn probability", f"{probabilities.mean():.1%}")
    st.dataframe(result, use_container_width=True)
    st.download_button("Download predictions CSV", result.to_csv(index=False),
                       file_name="churn_predictions.csv", mime="text/csv")
