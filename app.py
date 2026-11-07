import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

st.set_page_config(page_title="ML Classification Model Evaluation", page_icon="📊", layout="wide")
st.title("Machine Learning Classification Model Evaluation")
st.write(
    "Upload the held-out test CSV containing the 30 input features and the target column. "
    "The application loads the saved models and evaluates them without retraining."
)

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "target"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

uploaded = st.file_uploader("Upload test dataset (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload test_data.csv to view model evaluation results.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if TARGET not in data.columns:
    st.error("The uploaded CSV must contain the target column named 'target'.")
    st.stop()

if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

X_test = data.drop(columns=[TARGET])
y_test = data[TARGET]

model_name = st.selectbox("Select classification model", list(MODEL_FILES.keys()))
model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])

if not os.path.exists(model_path):
    st.error(f"Saved model file not found: {model_path}")
    st.stop()

model = load_model(model_path)
expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None

if expected_features is not None:
    missing = sorted(set(expected_features) - set(X_test.columns))
    extra = sorted(set(X_test.columns) - set(expected_features))
    if missing:
        st.error(f"Missing required feature columns: {', '.join(missing)}")
        st.stop()
    if extra:
        st.warning(f"Extra columns will be ignored: {', '.join(extra)}")
    X_test = X_test[expected_features]

try:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
except Exception as exc:
    st.error(f"Prediction failed. Check the uploaded feature names and values. Details: {exc}")
    st.stop()

labels = sorted(pd.Series(y_test).dropna().unique().tolist())
if len(labels) != 2:
    st.error("AUC calculation in this assignment app expects a binary classification target.")
    st.stop()

try:
    auc = roc_auc_score(y_test, probabilities)
except Exception:
    auc = float("nan")

results = {
    "Accuracy": accuracy_score(y_test, predictions),
    "AUC": auc,
    "Precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
    "Recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
    "F1 Score": f1_score(y_test, predictions, pos_label=1, zero_division=0),
    "MCC": matthews_corrcoef(y_test, predictions),
}

st.subheader("Evaluation Metrics")
cols = st.columns(3)
for i, (label, value) in enumerate(results.items()):
    cols[i % 3].metric(label, f"{value:.4f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, cmap="Blues")
ax.set_title(f"{model_name} - Confusion Matrix")
st.pyplot(fig)
plt.close(fig)

st.subheader("Classification Report")
report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader("Prediction Preview")
preview = X_test.copy()
preview["Actual Target"] = y_test.values
preview["Predicted Target"] = predictions
st.dataframe(preview.head(20), use_container_width=True)
