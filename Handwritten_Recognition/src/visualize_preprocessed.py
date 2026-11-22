
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import string

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "preprocessed"


# Load preprocessed datasets
mnist = np.load(DATA_PATH / "mnist_preprocessed.npz")
emnist = np.load(DATA_PATH / "emnist_preprocessed.npz")

x_mnist = mnist["x_train"]
y_mnist = mnist["y_train"]

x_emnist = emnist["x_train"]
y_emnist = emnist["y_train"]

letters = string.ascii_uppercase


# ---------------------------------------
# Visualize MNIST
# ---------------------------------------

plt.figure(figsize=(10, 6))

for i in range(12):
    plt.subplot(3, 4, i + 1)

    plt.imshow(x_mnist[i].squeeze(), cmap="gray")
    plt.title(f"Label: {y_mnist[i]}")
    plt.axis("off")

plt.suptitle("MNIST - Handwritten Digits")
plt.tight_layout()
plt.show()


# ---------------------------------------
# Visualize EMNIST
# ---------------------------------------

plt.figure(figsize=(10, 6))

for i in range(12):
    plt.subplot(3, 4, i + 1)

    plt.imshow(x_emnist[i].squeeze(), cmap="gray")
    plt.title(f"Label: {letters[y_emnist[i]]}")
    plt.axis("off")

plt.suptitle("EMNIST - Handwritten Letters")
plt.tight_layout()
plt.show()


print("MNIST image shape:", x_mnist.shape)
print("EMNIST image shape:", x_emnist.shape)

print("\nVisualization completed!")
