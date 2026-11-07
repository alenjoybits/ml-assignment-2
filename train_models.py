"""
Reproducible training script for Machine Learning Assignment 2.
Dataset: Breast Cancer Wisconsin (Diagnostic), packaged with scikit-learn.
Run: python train_models.py
This script creates the held-out test_data.csv, five saved models, and evaluation_metrics.csv.
"""
import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target  # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
test_data = X_test.copy()
test_data["target"] = y_test.values
test_data.to_csv(os.path.join(OUTPUT_DIR, "test_data.csv"), index=False)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(probability=True, random_state=RANDOM_STATE))
    ]),
}

rows = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
        "F1": f1_score(y_test, predictions, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    })
    filename = model_name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, filename))

pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))
