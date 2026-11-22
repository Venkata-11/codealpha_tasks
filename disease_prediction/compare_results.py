
import pandas as pd

# ==========================================
# 1. LOAD ALL ALGORITHM RESULTS
# ==========================================

result_files = [
    "results/logistic_regression_results.csv",
    "results/svm_results.csv",
    "results/random_forest_results.csv",
    "results/xgboost_results.csv"
]

all_results = []

for file in result_files:
    df = pd.read_csv(file)
    all_results.append(df)


# ==========================================
# 2. COMBINE ALL RESULTS
# ==========================================

comparison_df = pd.concat(
    all_results,
    ignore_index=True
)


# ==========================================
# 3. RANK MODELS USING F1 SCORE
# ==========================================

comparison_df["F1 Rank"] = (
    comparison_df.groupby("Dataset")["F1 Score"]
    .rank(method="min", ascending=False)
    .astype(int)
)


# ==========================================
# 4. SORT RESULTS
# ==========================================

comparison_df = comparison_df.sort_values(
    by=["Dataset", "F1 Score"],
    ascending=[True, False]
)


# ==========================================
# 5. DISPLAY FINAL COMPARISON
# ==========================================

print("\n========== COMPLETE MODEL COMPARISON ==========")

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2%}".format,
            "Precision": "{:.2%}".format,
            "Recall": "{:.2%}".format,
            "F1 Score": "{:.2%}".format
        }
    )
)


# ==========================================
# 6. SAVE CONSOLIDATED RESULTS
# ==========================================

comparison_df.to_csv(
    "results/all_model_comparison.csv",
    index=False
)

print("\nAll model results combined successfully!")
print("Saved at: results/all_model_comparison.csv")
