import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. CREATE REQUIRED FOLDERS
# ==========================================

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


# ==========================================
# 2. DEFINE THE DATASETS
# ==========================================

datasets = {
    "Heart Disease": "data/heart_disease/heart_processed.npz",
    "Diabetes": "data/diabetes/diabetes_processed.npz",
    "Breast Cancer": "data/breast_cancer/breast_processed.npz"
}


# ==========================================
# 3. COMMON MODEL EVALUATION FUNCTION
# ==========================================

def evaluate_model(dataset_name, y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)

    # All three datasets now use binary classification.

    precision = precision_score(
        y_test,
        y_pred,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="binary",
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    return accuracy, precision, recall, f1, cm


# ==========================================
# 4. COMMON FUNCTION TO SAVE RESULTS
# ==========================================

def save_results(results, filename, algorithm_name):

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        f"results/{filename}",
        index=False
    )

    print(f"\n========== {algorithm_name} SUMMARY ==========")

    print(results_df.to_string(index=False))

    print("\nResults saved successfully!")


# ==========================================
# 5. TRAIN LOGISTIC REGRESSION
# ==========================================

logistic_results = []

for dataset_name, file_path in datasets.items():

    print("\n==========================================")
    print("DATASET:", dataset_name)
    print("ALGORITHM: LOGISTIC REGRESSION")
    print("==========================================")

    data = np.load(file_path)

    X_train = data["X_train"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\nModel training completed!")

    y_pred = model.predict(X_test)

    accuracy, precision, recall, f1, cm = evaluate_model(
        dataset_name, y_test, y_pred
    )

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    model_filename = (
        dataset_name.lower().replace(" ", "_")
        + "_logistic_regression.pkl"
    )

    model_path = Path("models") / model_filename

    joblib.dump(model, model_path)

    print("\nModel saved at:", model_path)

    logistic_results.append({
        "Dataset": dataset_name,
        "Algorithm": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


save_results(
    logistic_results,
    "logistic_regression_results.csv",
    "LOGISTIC REGRESSION"
)


# ==========================================
# 6. TRAIN SUPPORT VECTOR MACHINE (SVM)
# ==========================================

svm_results = []

for dataset_name, file_path in datasets.items():

    print("\n==========================================")
    print("DATASET:", dataset_name)
    print("ALGORITHM: SUPPORT VECTOR MACHINE")
    print("==========================================")

    data = np.load(file_path)

    X_train = data["X_train"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )

    model.fit(X_train, y_train)

    print("\nModel training completed!")

    y_pred = model.predict(X_test)

    accuracy, precision, recall, f1, cm = evaluate_model(
        dataset_name, y_test, y_pred
    )

    print("\n========== SVM MODEL EVALUATION ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    model_filename = (
        dataset_name.lower().replace(" ", "_")
        + "_svm.pkl"
    )

    model_path = Path("models") / model_filename

    joblib.dump(model, model_path)

    print("\nModel saved at:", model_path)

    svm_results.append({
        "Dataset": dataset_name,
        "Algorithm": "SVM",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


save_results(
    svm_results,
    "svm_results.csv",
    "SVM"
)


# ==========================================
# 7. TRAIN RANDOM FOREST
# ==========================================

rf_results = []

for dataset_name, file_path in datasets.items():

    print("\n==========================================")
    print("DATASET:", dataset_name)
    print("ALGORITHM: RANDOM FOREST")
    print("==========================================")

    data = np.load(file_path)

    X_train = data["X_train"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("\nModel training completed!")

    y_pred = model.predict(X_test)

    accuracy, precision, recall, f1, cm = evaluate_model(
        dataset_name, y_test, y_pred
    )

    print("\n========== RANDOM FOREST EVALUATION ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    model_filename = (
        dataset_name.lower().replace(" ", "_")
        + "_random_forest.pkl"
    )

    model_path = Path("models") / model_filename

    joblib.dump(model, model_path)

    print("\nModel saved at:", model_path)

    rf_results.append({
        "Dataset": dataset_name,
        "Algorithm": "Random Forest",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


save_results(
    rf_results,
    "random_forest_results.csv",
    "RANDOM FOREST"
)


# ==========================================
# 8. TRAIN XGBOOST
# ==========================================

xgb_results = []

for dataset_name, file_path in datasets.items():

    print("\n==========================================")
    print("DATASET:", dataset_name)
    print("ALGORITHM: XGBOOST")
    print("==========================================")

    data = np.load(file_path)

    X_train = data["X_train"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    # Binary classification configuration for all datasets.

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("\nModel training completed!")

    y_pred = model.predict(X_test)

    accuracy, precision, recall, f1, cm = evaluate_model(
        dataset_name, y_test, y_pred
    )

    print("\n========== XGBOOST EVALUATION ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    model_filename = (
        dataset_name.lower().replace(" ", "_")
        + "_xgboost.pkl"
    )

    model_path = Path("models") / model_filename

    joblib.dump(model, model_path)

    print("\nModel saved at:", model_path)

    xgb_results.append({
        "Dataset": dataset_name,
        "Algorithm": "XGBoost",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


save_results(
    xgb_results,
    "xgboost_results.csv",
    "XGBOOST"
)


# ==========================================
# 9. TRAINING COMPLETED
# ==========================================

print("\n==========================================")
print("ALL MODEL TRAINING COMPLETED!")
print("==========================================")

print("\nTotal Models Trained: 12")
print("Heart Disease: Binary Classification")
print("Diabetes: Binary Classification")
print("Breast Cancer: Binary Classification")

print("\nAll trained models are saved in the models folder.")
print("All evaluation results are saved in the results folder.")