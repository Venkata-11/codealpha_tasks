
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load test data
data = np.load("data/preprocessed/emnist_stratified_splits.npz")

x_test = data["x_test"]
y_test = data["y_test"]

# Load trained model
model = tf.keras.models.load_model("models/emnist_cnn.keras")

# Predict test images
probabilities = model.predict(x_test, verbose=1)
predictions = np.argmax(probabilities, axis=1)

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Find correct and incorrect predictions
correct_indices = np.where(predictions == y_test)[0]
incorrect_indices = np.where(predictions != y_test)[0]

print("Correct predictions:", len(correct_indices))
print("Incorrect predictions:", len(incorrect_indices))

# Display 10 correct and 10 incorrect examples
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

fig, axes = plt.subplots(4, 5, figsize=(12, 10))

for ax, index in zip(axes.flatten(), selected_indices):

    image = x_test[index].squeeze()

    actual = letters[y_test[index]]
    predicted = letters[predictions[index]]
    confidence = probabilities[index][predictions[index]] * 100

    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"Actual: {actual}\nPredicted: {predicted}\nConfidence: {confidence:.1f}%"
    )
    ax.axis("off")

plt.suptitle("EMNIST CNN - Sample Predictions")
plt.tight_layout()
plt.savefig("results/emnist_sample_predictions.png", dpi=300)
plt.show()

print("\nPrediction visualisation saved successfully!")
