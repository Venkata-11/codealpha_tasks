import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Heart Disease dataset
heart_df = pd.read_csv("data/heart_disease/heart_disease_raw.csv")

# # Display the first five rows
# print("FIRST FIVE ROWS")
# print(heart_df.head())

# # Display dataset dimensions
# print("\nDATASET SHAPE")
# print(heart_df.shape)

# # Display column names and data types
# print("\nDATASET INFORMATION")
# print(heart_df.info())

# # Display statistical summary
# print("\nSTATISTICAL SUMMARY")
# print(heart_df.describe())

# # Check missing values
# print("\nMISSING VALUES")
# print(heart_df.isnull().sum())

# # Check duplicate records
# print("\nDUPLICATE RECORDS")
# print(heart_df.duplicated().sum())

# # Check target distribution
# print("\nTARGET DISTRIBUTION")
# print(heart_df["num"].value_counts().sort_index())



# # ==========================================
# # DIABETES DATASET - EDA
# # ==========================================

diabetes_df = pd.read_csv("data/diabetes/diabetes_raw.csv")

# print("\n\n========== DIABETES DATASET ==========")

# # First five rows
# print("\nFIRST FIVE ROWS")
# print(diabetes_df.head())

# # Dataset dimensions
# print("\nDATASET SHAPE")
# print(diabetes_df.shape)

# # Dataset information
# print("\nDATASET INFORMATION")
# diabetes_df.info()

# # Statistical summary
# print("\nSTATISTICAL SUMMARY")
# print(diabetes_df.describe())

# # Missing values
# print("\nMISSING VALUES")
# print(diabetes_df.isnull().sum())

# # Duplicate records
# print("\nDUPLICATE RECORDS")
# print(diabetes_df.duplicated().sum())

# # Target distribution
# print("\nTARGET DISTRIBUTION")
# print(diabetes_df["class"].value_counts())

# # Investigate duplicate records
# print("\nDUPLICATE RECORD EXAMPLES")
# print(diabetes_df[diabetes_df.duplicated()].head(10))

# # Count unique records
# print("\nUNIQUE RECORD COUNT")
# print(len(diabetes_df.drop_duplicates()))



# # ==========================================
# # BREAST CANCER DATASET - EDA
# # ==========================================

breast_df = pd.read_csv(
    "data/breast_cancer/breast_cancer_raw.csv"
)

# print("\n\n========== BREAST CANCER DATASET ==========")

# # First five rows
# print("\nFIRST FIVE ROWS")
# print(breast_df.head())

# # Dataset dimensions
# print("\nDATASET SHAPE")
# print(breast_df.shape)

# # Dataset information
# print("\nDATASET INFORMATION")
# breast_df.info()

# # Statistical summary
# print("\nSTATISTICAL SUMMARY")
# print(breast_df.describe())

# # Missing values
# print("\nMISSING VALUES")
# print(breast_df.isnull().sum())

# # Duplicate records
# print("\nDUPLICATE RECORDS")
# print(breast_df.duplicated().sum())

# # Target distribution
# print("\nTARGET DISTRIBUTION")
# print(breast_df["Diagnosis"].value_counts())


# # ==========================================
# # TARGET DISTRIBUTION VISUALIZATIONS
# # ==========================================

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # Heart Disease
# sns.countplot(
#     data=heart_df,
#     x="num",
#     ax=axes[0]
# )
# axes[0].set_title("Heart Disease Target Distribution")
# axes[0].set_xlabel("Disease Category")
# axes[0].set_ylabel("Number of Records")

# # Diabetes
# sns.countplot(
#     data=diabetes_df,
#     x="class",
#     ax=axes[1]
# )
# axes[1].set_title("Diabetes Target Distribution")
# axes[1].set_xlabel("Diabetes Class")
# axes[1].set_ylabel("Number of Records")

# # Breast Cancer
# sns.countplot(
#     data=breast_df,
#     x="Diagnosis",
#     ax=axes[2]
# )
# axes[2].set_title("Breast Cancer Target Distribution")
# axes[2].set_xlabel("Diagnosis")
# axes[2].set_ylabel("Number of Records")

# plt.tight_layout()
# plt.show()



# # ==========================================
# # FEATURE DISTRIBUTION ANALYSIS
# # ==========================================

# import matplotlib.pyplot as plt
# import seaborn as sns

# # ==========================================
# # 1. HEART DISEASE - FEATURE DISTRIBUTIONS
# # ==========================================

# heart_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# heart_df[heart_features].hist(
#     figsize=(12, 8),
#     bins=15,
#     edgecolor="black"
# )

# plt.suptitle("Heart Disease - Feature Distributions")
# plt.tight_layout()
# plt.show()


# # ==========================================
# # 2. DIABETES - FEATURE DISTRIBUTIONS
# # ==========================================

# plt.figure(figsize=(8, 5))

# sns.histplot(
#     data=diabetes_df,
#     x="age",
#     bins=15,
#     kde=True
# )

# plt.title("Diabetes - Age Distribution")
# plt.xlabel("Age")
# plt.ylabel("Number of Records")
# plt.tight_layout()
# plt.show()


# # Visualize selected diabetes symptoms

# diabetes_symptoms = [
#     "polyuria",
#     "polydipsia",
#     "sudden_weight_loss"
# ]

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# for i, feature in enumerate(diabetes_symptoms):
#     sns.countplot(
#         data=diabetes_df,
#         x=feature,
#         ax=axes[i]
#     )
#     axes[i].set_title(feature.replace("_", " ").title())
#     axes[i].set_xlabel("Response")
#     axes[i].set_ylabel("Number of Records")

# plt.suptitle("Diabetes - Symptom Distributions")
# plt.tight_layout()
# plt.show()


# # ==========================================
# # 3. BREAST CANCER - FEATURE DISTRIBUTIONS
# # ==========================================

# breast_features = [
#     "radius1",
#     "texture1",
#     "perimeter1",
#     "area1",
#     "smoothness1",
#     "concavity1"
# ]

# breast_df[breast_features].hist(
#     figsize=(12, 8),
#     bins=15,
#     edgecolor="black"
# )

# plt.suptitle("Breast Cancer - Feature Distributions")
# plt.tight_layout()
# plt.show()



# ==========================================
# BOXPLOT ANALYSIS
# ==========================================

# ==========================================
# 1. HEART DISEASE BOXPLOTS
# ==========================================

# Create a temporary binary target for visualization.
# Original 0 = No disease; original 1-4 = Disease.
# heart_df["disease_status"] = heart_df["num"].apply(
#     lambda value: "Disease" if value > 0 else "No Disease"
# )

# heart_features_box = ["age", "chol", "trestbps"]

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# for i, feature in enumerate(heart_features_box):
#     sns.boxplot(
#         data=heart_df,
#         x="disease_status",
#         y=feature,
#         ax=axes[i]
#     )
#     axes[i].set_title(f"{feature} by Disease Status")

# plt.suptitle("Heart Disease - Boxplot Analysis")
# plt.tight_layout()
# plt.show()


# # ==========================================
# # 2. DIABETES BOXPLOTS
# # ==========================================

# diabetes_features_box = ["age"]

# fig, axes = plt.subplots(1, 1, figsize=(7, 5))

# sns.boxplot(
#     data=diabetes_df,
#     x="class",
#     y="age",
#     ax=axes
# )

# axes.set_title("Diabetes - Age by Class")
# plt.tight_layout()
# plt.show()


# # ==========================================
# # 3. BREAST CANCER BOXPLOTS
# # ==========================================

# breast_features_box = ["radius1", "texture1", "area1"]

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# for i, feature in enumerate(breast_features_box):
#     sns.boxplot(
#         data=breast_df,
#         x="Diagnosis",
#         y=feature,
#         ax=axes[i]
#     )
#     axes[i].set_title(f"{feature} by Diagnosis")

# plt.suptitle("Breast Cancer - Boxplot Analysis")
# plt.tight_layout()
# plt.show()



# ==========================================
# CORRELATION ANALYSIS
# ==========================================

# ==========================================
# 1. HEART DISEASE CORRELATION
# ==========================================

heart_corr = heart_df.copy()

# Convert original target into binary form
# 0 = No Disease, 1 = Disease
heart_corr["target_binary"] = (
    heart_corr["num"] > 0
).astype(int)

heart_corr_matrix = heart_corr.drop(
    columns=["num"]
).corr(numeric_only=True)

plt.figure(figsize=(12, 9))

sns.heatmap(
    heart_corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Heart Disease - Correlation Heatmap")
plt.tight_layout()
plt.show()


# ==========================================
# 2. DIABETES CORRELATION
# ==========================================

diabetes_corr = diabetes_df.copy()

# Convert categorical values into numerical values
diabetes_mapping = {
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0,
    "Positive": 1,
    "Negative": 0
}

for column in diabetes_corr.select_dtypes(
    include=["object", "str"]
).columns:
    diabetes_corr[column] = diabetes_corr[column].map(
        diabetes_mapping
    )

diabetes_corr_matrix = diabetes_corr.corr()

plt.figure(figsize=(14, 11))

sns.heatmap(
    diabetes_corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Diabetes - Correlation Heatmap")
plt.tight_layout()
plt.show()


# ==========================================
# 3. BREAST CANCER CORRELATION
# ==========================================

breast_corr = breast_df.copy()

# Convert diagnosis into numerical values
# B = 0 (Benign), M = 1 (Malignant)
breast_corr["target_binary"] = (
    breast_corr["Diagnosis"].map({
        "B": 0,
        "M": 1
    })
)

# Calculate correlation with the target
breast_target_corr = breast_corr.drop(
    columns=["Diagnosis"]
).corr()["target_binary"].drop(
    "target_binary"
)

# Select the 10 features with the strongest
# absolute correlation with the target
top_breast_features = (
    breast_target_corr.abs()
    .sort_values(ascending=False)
    .head(10)
    .index
)

breast_corr_matrix = breast_corr[
    list(top_breast_features) + ["target_binary"]
].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    breast_corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Breast Cancer - Top 10 Feature Correlations")
plt.tight_layout()
plt.show()
