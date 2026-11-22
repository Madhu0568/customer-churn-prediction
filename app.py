"""Streamlit application for batch customer churn prediction."""
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

import joblib
import pandas as pd
import streamlit as st

from src.data_preprocessing import engineer_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
TRAINING_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)


def prepare_model() -> None:
    """Download the public training data and train a model on first cloud startup."""
    if MODEL_PATH.is_file():
        return

    st.info(
        "First-time setup: downloading the public IBM Telco sample dataset "
        "and training the prediction model. This can take a few minutes."
    )
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".csv", prefix="telco_training_", delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        urllib.request.urlretrieve(TRAINING_DATA_URL, temp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.train",
                "--data",
                str(temp_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Model training failed during app setup.\n"
                + result.stdout[-4000:]
                + "\n"
                + result.stderr[-4000:]
            )
    finally:
        temp_path.unlink(missing_ok=True)

    if not MODEL_PATH.is_file():
        raise RuntimeError("Training finished but the model file was not created.")


@st.cache_resource
def load_model():
    prepare_model()
    return joblib.load(MODEL_PATH)


st.title("Customer Churn Prediction")
st.write(
    "Upload telecom customer records to estimate churn probabilities "
    "using a machine-learning model."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to prepare the prediction model: {error}")
    st.stop()

uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        features = engineer_features(data)
        features = features.drop(
            columns=["Churn", "customerID"],
            errors="ignore",
        )

        probabilities = model.predict_proba(features)[:, 1]
        labels = model.predict(features)

        result = data.copy()
        result["churn_prediction"] = [
            "Yes" if value == 1 else "No" for value in labels
        ]
        result["churn_probability"] = probabilities

        st.subheader("Prediction summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Customers", len(result))
        col2.metric("Predicted churn", int((labels == 1).sum()))
        col3.metric(
            "Average churn probability",
            f"{probabilities.mean():.1%}",
        )

        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            result.to_csv(index=False),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
