from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# Download the Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Separate features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine features and target into one table
heart_df = pd.concat([X, y], axis=1)

# Create the file path
save_path = Path("data/heart_disease/heart_disease_raw.csv")

# Save the dataset as a CSV file
heart_df.to_csv(save_path, index=False)

print("Heart Disease dataset downloaded and saved successfully!")
print("Dataset Shape:", heart_df.shape)
print("Saved Location:", save_path)

# Download the Early Stage Diabetes dataset
diabetes = fetch_ucirepo(id=529)

# Separate features and target
X_diabetes = diabetes.data.features
y_diabetes = diabetes.data.targets

# Combine features and target
diabetes_df = pd.concat([X_diabetes, y_diabetes], axis=1)

# Save the dataset
diabetes_path = Path("data/diabetes/diabetes_raw.csv")

diabetes_df.to_csv(diabetes_path, index=False)

print("\nDiabetes Dataset")
print("----------------")

print("Dataset Shape:", diabetes_df.shape)
print("Saved Location:", diabetes_path)

print("\nFirst five rows:")
print(diabetes_df.head())


# Download the Breast Cancer Wisconsin dataset
breast_cancer = fetch_ucirepo(id=17)

# Separate features and target
X_breast = breast_cancer.data.features
y_breast = breast_cancer.data.targets

# Combine features and target
breast_df = pd.concat([X_breast, y_breast], axis=1)

# Save the dataset
breast_path = Path("data/breast_cancer/breast_cancer_raw.csv")

breast_df.to_csv(breast_path, index=False)

print("\nBreast Cancer Dataset")
print("---------------------")

print("Dataset Shape:", breast_df.shape)
print("Saved Location:", breast_path)

print("\nFirst five rows:")
print(breast_df.head())