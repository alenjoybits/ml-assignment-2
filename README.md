# Machine Learning Assignment 2 – Classification Models

## a. Problem statement

The objective is to implement five classification algorithms on one common dataset and compare their performance using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The trained models are also made available through an interactive Streamlit application that evaluates an uploaded test CSV without retraining.

## b. Dataset description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (also distributed in the scikit-learn dataset collection).
- **Task:** Binary classification of breast-cancer samples as malignant or benign.
- **Instances:** 569.
- **Input features:** 30 numeric features.
- **Target:** `target` (0 = malignant, 1 = benign).
- **Train/test split:** 80% training and 20% held-out testing.
- **Split configuration:** stratified split with `random_state=42`.
- **Preprocessing:** StandardScaler is used within pipelines for Logistic Regression, KNN, and SVM. Decision Tree and Gaussian Naive Bayes are trained without feature scaling.
- **Test file:** `test_data.csv` contains the held-out test features and target column.

## c. Github Repository Link

**GitHub Repository:** https://github.com/alenjoybits/ml-assignment-2

The repository contains `app.py`, `requirements.txt`, this `README.md`, `test_data.csv`, `evaluation_metrics.csv`, and the five saved model files under `model/`.

## d. Models used

The five required classifiers were trained on the same training split and evaluated on the same held-out test split.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |\n| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |\n| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |\n| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n
### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 0.9825 accuracy, 0.9954 AUC, and 0.9861 for precision, recall, and F1. Its MCC of 0.9623 indicates very strong agreement between predictions and actual classes. It is one of the two strongest models on this test set. |
| Decision Tree | Recorded the lowest accuracy (0.9123), AUC (0.9157), F1 (0.9286), and MCC (0.8174) among the five models. It performs reasonably well but is weaker than the other classifiers on this held-out test set. |
| KNN | Achieved 0.9561 accuracy and 0.9788 AUC. Recall (0.9722) is higher than precision (0.9589), indicating that it identifies a high proportion of benign samples while making somewhat more false-positive predictions than a model with higher precision. |
| Naive Bayes | Achieved 0.9386 accuracy and 0.9878 AUC. Its AUC is the second highest, showing strong class-ranking performance, although its accuracy, F1 (0.9517), and MCC (0.8676) are lower than Logistic Regression and SVM. |
| Support Vector Machine | Matched Logistic Regression on accuracy (0.9825), precision (0.9861), recall (0.9861), F1 (0.9861), and MCC (0.9623). Its AUC (0.9950) is marginally below Logistic Regression's 0.9954, so the two models are nearly tied. |
| **Overall, Winner for your dataset?** | **Logistic Regression is selected as the overall winner. It ties with Support Vector Machine for the highest F1 score (0.9861) and matches it on accuracy, precision, recall, and MCC, while achieving a slightly higher AUC (0.9954 versus 0.9950).** |

## Streamlit application

The application provides:
1. CSV upload for the held-out test data.
2. A dropdown to select one of the five saved classification models.
3. Display of Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
4. A confusion matrix and classification report.

The uploaded CSV must contain all required input feature columns and the `target` column. The app loads the saved model and does not train or retrain a model on uploaded data.

**Live Streamlit App:** https://ml-assignment-2-dt9xypy89wuc7amfkmqucg.streamlit.app/

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` in the app to reproduce the evaluation.

## Repository structure

```text
ML_Assignment_2_Solution/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── evaluation_metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```
