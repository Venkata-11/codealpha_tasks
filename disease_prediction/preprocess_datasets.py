# ==========================================
# HEART DISEASE - DATA PREPROCESSING
# ==========================================

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ==========================================
# 1. LOAD DATASET
# ==========================================

heart_df = pd.read_csv(
    "data/heart_disease/heart_disease_raw.csv"
)

print("Original Dataset Shape:", heart_df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================


# Original target:
# 0 = No heart disease
# 1, 2, 3, 4 = Different levels of heart disease

# Keep all five original classes for multiclass classification.

X = heart_df.drop(columns=["num"])
y = (heart_df["num"] > 0).astype(int)


print("\nInput Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())


# ==========================================
# 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ==========================================

numerical_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


# ==========================================
# 4. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)


# ==========================================
# 5. CREATE PREPROCESSING PIPELINES
# ==========================================

# Numerical features:
# Fill missing values using median.
# Scale values using StandardScaler.

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical features:
# Fill missing values using most frequent value.
# Convert categories into numerical columns.

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])


# Combine both pipelines.

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==========================================
# 6. APPLY PREPROCESSING
# ==========================================

# Fit only on training data.
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the already-fitted transformations to test data.
X_test_processed = preprocessor.transform(X_test)


# ==========================================
# 7. CHECK RESULTS
# ==========================================

print("\n========== PREPROCESSING RESULTS ==========")

print("Processed Training Shape:", X_train_processed.shape)
print("Processed Testing Shape:", X_test_processed.shape)

print("\nMissing Values Before Imputation:")
print(X_train.isnull().sum())

print("\nMissing Values After Imputation:")

print(
    "Training:",
    pd.DataFrame(X_train_processed).isnull().sum().sum()
)

print(
    "Testing:",
    pd.DataFrame(X_test_processed).isnull().sum().sum()
)


# ==========================================
# 8. SAVE PREPROCESSOR
# ==========================================

joblib.dump(
    preprocessor,
    "data/heart_disease/heart_preprocessor.pkl"
)

np.savez_compressed(
    "data/heart_disease/heart_processed.npz",
    X_train=X_train_processed,
    X_test=X_test_processed,
    y_train=y_train.to_numpy(),
    y_test=y_test.to_numpy()
)

print("Processed training and testing data saved!")

print("\nHeart Disease preprocessing completed!")
print("Preprocessor saved successfully.")




# ==========================================
# DIABETES - DATA PREPROCESSING
# ==========================================

# ==========================================
# 1. LOAD DATASET
# ==========================================

diabetes_df = pd.read_csv(
    "data/diabetes/diabetes_raw.csv"
)

print("\n\n========== DIABETES PREPROCESSING ==========")

print("Original Dataset Shape:", diabetes_df.shape)


# ==========================================
# 2. INVESTIGATE AND HANDLE DUPLICATES
# ==========================================

print("\nDuplicate Records Before Cleaning:")
print(diabetes_df.duplicated().sum())

# Create a working copy and remove exact duplicates.
# The original CSV remains unchanged.

diabetes_clean = diabetes_df.drop_duplicates().copy()

print("\nDataset Shape After Removing Duplicates:")
print(diabetes_clean.shape)

print("\nDuplicate Records After Cleaning:")
print(diabetes_clean.duplicated().sum())


# ==========================================
# 3. CLEAN CATEGORICAL TEXT
# ==========================================

# Remove accidental spaces from text values.

for column in diabetes_clean.select_dtypes(
    include=["object", "str"]
).columns:
    diabetes_clean[column] = (
        diabetes_clean[column].str.strip()
    )


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

# Target:
# Positive = 1
# Negative = 0

diabetes_clean["class"] = diabetes_clean["class"].map({
    "Positive": 1,
    "Negative": 0
})

X_diabetes = diabetes_clean.drop(columns=["class"])
y_diabetes = diabetes_clean["class"]

print("\nInput Features Shape:", X_diabetes.shape)
print("Target Shape:", y_diabetes.shape)

print("\nTarget Distribution:")
print(y_diabetes.value_counts())


# ==========================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ==========================================

numerical_features_diabetes = ["age"]

categorical_features_diabetes = [
    column for column in X_diabetes.columns
    if column not in numerical_features_diabetes
]

print("\nNumerical Features:")
print(numerical_features_diabetes)

print("\nCategorical Features:")
print(categorical_features_diabetes)


# ==========================================
# 6. SPLIT INTO TRAINING AND TESTING DATA
# ==========================================

X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(
    X_diabetes,
    y_diabetes,
    test_size=0.20,
    random_state=42,
    stratify=y_diabetes
)

print("\nTraining Features:", X_train_diabetes.shape)
print("Testing Features:", X_test_diabetes.shape)


# ==========================================
# 7. CREATE PREPROCESSING PIPELINES
# ==========================================

# Numerical pipeline

numerical_pipeline_diabetes = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical pipeline

categorical_pipeline_diabetes = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])


# Combine pipelines

preprocessor_diabetes = ColumnTransformer([
    (
        "numerical",
        numerical_pipeline_diabetes,
        numerical_features_diabetes
    ),
    (
        "categorical",
        categorical_pipeline_diabetes,
        categorical_features_diabetes
    )
])


# ==========================================
# 8. APPLY PREPROCESSING
# ==========================================

X_train_diabetes_processed = (
    preprocessor_diabetes.fit_transform(X_train_diabetes)
)

X_test_diabetes_processed = (
    preprocessor_diabetes.transform(X_test_diabetes)
)


# ==========================================
# 9. CHECK RESULTS
# ==========================================

print("\n========== DIABETES PREPROCESSING RESULTS ==========")

print(
    "Processed Training Shape:",
    X_train_diabetes_processed.shape
)

print(
    "Processed Testing Shape:",
    X_test_diabetes_processed.shape
)

print("\nMissing Values Before Imputation:")
print(X_train_diabetes.isnull().sum())

print("\nMissing Values After Imputation:")

print(
    "Training:",
    pd.DataFrame(X_train_diabetes_processed).isnull().sum().sum()
)

print(
    "Testing:",
    pd.DataFrame(X_test_diabetes_processed).isnull().sum().sum()
)


# ==========================================
# 10. SAVE PREPROCESSOR
# ==========================================

joblib.dump(
    preprocessor_diabetes,
    "data/diabetes/diabetes_preprocessor.pkl"
)

np.savez_compressed(
    "data/diabetes/diabetes_processed.npz",
    X_train=X_train_diabetes_processed,
    X_test=X_test_diabetes_processed,
    y_train=y_train_diabetes.to_numpy(),
    y_test=y_test_diabetes.to_numpy()
)

print("Processed training and testing data saved!")

print("\nDiabetes preprocessing completed!")
print("Preprocessor saved successfully.")



# ==========================================
# BREAST CANCER - DATA PREPROCESSING
# ==========================================

# ==========================================
# 1. LOAD DATASET
# ==========================================

breast_df = pd.read_csv(
    "data/breast_cancer/breast_cancer_raw.csv"
)

print("\n\n========== BREAST CANCER PREPROCESSING ==========")

print("Original Dataset Shape:", breast_df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

# Diagnosis:
# B = 0 (Benign)
# M = 1 (Malignant)

breast_df["Diagnosis"] = breast_df["Diagnosis"].map({
    "B": 0,
    "M": 1
})

X_breast = breast_df.drop(columns=["Diagnosis"])
y_breast = breast_df["Diagnosis"]

print("\nInput Features Shape:", X_breast.shape)
print("Target Shape:", y_breast.shape)

print("\nTarget Distribution:")
print(y_breast.value_counts())


# ==========================================
# 3. IDENTIFY NUMERICAL FEATURES
# ==========================================

numerical_features_breast = X_breast.columns.tolist()

print("\nNumber of Numerical Features:")
print(len(numerical_features_breast))


# ==========================================
# 4. SPLIT INTO TRAINING AND TESTING DATA
# ==========================================

X_train_breast, X_test_breast, y_train_breast, y_test_breast = train_test_split(
    X_breast,
    y_breast,
    test_size=0.20,
    random_state=42,
    stratify=y_breast
)

print("\nTraining Features:", X_train_breast.shape)
print("Testing Features:", X_test_breast.shape)


# ==========================================
# 5. CREATE PREPROCESSING PIPELINE
# ==========================================

breast_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ==========================================
# 6. APPLY PREPROCESSING
# ==========================================

X_train_breast_processed = breast_pipeline.fit_transform(
    X_train_breast
)

X_test_breast_processed = breast_pipeline.transform(
    X_test_breast
)


# ==========================================
# 7. CHECK RESULTS
# ==========================================

print("\n========== BREAST CANCER PREPROCESSING RESULTS ==========")

print(
    "Processed Training Shape:",
    X_train_breast_processed.shape
)

print(
    "Processed Testing Shape:",
    X_test_breast_processed.shape
)

print("\nMissing Values Before Imputation:")
print(X_train_breast.isnull().sum().sum())

print("\nMissing Values After Imputation:")

print(
    "Training:",
    pd.DataFrame(X_train_breast_processed).isnull().sum().sum()
)

print(
    "Testing:",
    pd.DataFrame(X_test_breast_processed).isnull().sum().sum()
)


# ==========================================
# 8. SAVE PREPROCESSOR
# ==========================================

joblib.dump(
    breast_pipeline,
    "data/breast_cancer/breast_preprocessor.pkl"
)

np.savez_compressed(
    "data/breast_cancer/breast_processed.npz",
    X_train=X_train_breast_processed,
    X_test=X_test_breast_processed,
    y_train=y_train_breast.to_numpy(),
    y_test=y_test_breast.to_numpy()
)

print("Processed training and testing data saved!")

print("\nBreast Cancer preprocessing completed!")
print("Preprocessor saved successfully.")
