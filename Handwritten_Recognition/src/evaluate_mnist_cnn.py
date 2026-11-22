
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

print("========== MNIST MODEL EVALUATION ==========")

# Paths
DATA_PATH = Path("data/preprocessed/mnist_preprocessed.npz")
MODEL_PATH = Path("models/mnist_cnn.keras")
RESULTS_PATH = Path("results")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Load test dataset
data = np.load(DATA_PATH)

x_test = data["x_test"]
y_test = data["y_test"]

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel and test dataset loaded successfully!")

# Evaluate model
test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=1
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
predictions = model.predict(x_test, verbose=1)
y_pred = np.argmax(predictions, axis=1)

# Classification report
report = classification_report(
    y_test,
    y_pred,
    labels=list(range(10)),
    target_names=[str(i) for i in range(10)],
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

# Save report
with open(RESULTS_PATH / "mnist_classification_report.txt", "w") as file:
    file.write(report)

# Generate confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(10))
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.title("MNIST CNN - Confusion Matrix")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.tight_layout()

plt.savefig(
    RESULTS_PATH / "mnist_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved successfully!")

print("\n========== EVALUATION COMPLETED ==========")
