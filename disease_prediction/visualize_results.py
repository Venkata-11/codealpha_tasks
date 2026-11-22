import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the combined model comparison results
df = pd.read_csv("results/all_model_comparison.csv")

# Create a folder for charts
import os
os.makedirs("results/plots", exist_ok=True)

# Metrics to compare
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

# Set chart appearance
sns.set_theme(style="whitegrid")

# Generate one chart for each metric
for metric in metrics:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="Dataset",
        y=metric,
        hue="Algorithm"
    )

    plt.title(f"Model Comparison - {metric}")
    plt.ylabel(metric)
    plt.xlabel("Dataset")
    plt.ylim(0, 1.05)

    plt.legend(
        title="Algorithm",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    # Save chart
    filename = metric.lower().replace(" ", "_")

    plt.savefig(
        f"results/plots/{filename}_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

print("\nAll comparison charts generated successfully!")