
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load test data
data = np.load("data/preprocessed/mnist_preprocessed.npz")

x_test = data["x_test"]
y_test = data["y_test"]

# Load trained model
model = tf.keras.models.load_model("models/mnist_cnn.keras")

print("========== MNIST SAMPLE PREDICTIONS ==========")

# Generate predictions
probabilities = model.predict(x_test, verbose=1)
predictions = np.argmax(probabilities, axis=1)

# Find correct and incorrect predictions
correct_indices = np.where(predictions == y_test)[0]
incorrect_indices = np.where(predictions != y_test)[0]

print("\nCorrect predictions:", len(correct_indices))
print("Incorrect predictions:", len(incorrect_indices))

# Select 10 correct and 10 incorrect examples
np.random.seed(42)

selected_correct = np.random.choice(
    correct_indices,
    size=10,
    replace=False
)

selected_incorrect = np.random.choice(
    incorrect_indices,
    size=10,
    replace=False
)

selected_indices = np.concatenate([
    selected_correct,
    selected_incorrect
])

# Display predictions
fig, axes = plt.subplots(4, 5, figsize=(15, 12))

for ax, index in zip(axes.flatten(), selected_indices):

    image = x_test[index].squeeze()

    actual = y_test[index]
    predicted = predictions[index]

    confidence = probabilities[index][predicted] * 100

    ax.imshow(image, cmap="gray")

    # Identify whether prediction is correct
    status = "Correct" if actual == predicted else "Incorrect"

    ax.set_title(
        f"Actual: {actual} | Predicted: {predicted}\n"
        f"Confidence: {confidence:.1f}% | {status}",
        fontsize=9,
        pad=8
    )

    ax.axis("off")

# Main title
plt.suptitle(
    "MNIST CNN - Sample Predictions",
    fontsize=16
)

# Adjust spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save visualisation
plt.savefig(
    "results/mnist_sample_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

# Display visualisation
plt.show()

print("\nPrediction visualisation saved successfully!")
print("\n========== PREDICTION COMPLETED ==========")
