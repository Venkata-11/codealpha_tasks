
from pathlib import Path
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Locate the main project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

MNIST_PATH = PROJECT_DIR / "data" / "mnist"
EMNIST_PATH = PROJECT_DIR / "data" / "emnist"


def explore_dataset(dataset_name, data_path, dataset_id):
    print(f"\n========== {dataset_name} EXPLORATION ==========")

    # Load training and testing datasets
    (train_data, test_data), info = tfds.load(
        dataset_id,
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        data_dir=str(data_path)
    )

    print("\nDATASET INFORMATION")
    print("Training images:", info.splits["train"].num_examples)
    print("Testing images:", info.splits["test"].num_examples)
    print("Image shape:", info.features["image"].shape)
    print("Number of classes:", info.features["label"].num_classes)

    # Collect training labels and sample images
    label_counts = Counter()
    sample_images = []
    sample_labels = []

    for image, label in tfds.as_numpy(train_data):
        label_value = int(label)
        label_counts[label_value] += 1

        if len(sample_images) < 12:
            sample_images.append(image.squeeze())
            sample_labels.append(label_value)

    print("\nLABEL DISTRIBUTION")
    for label, count in sorted(label_counts.items()):
        print(f"Label {label}: {count} images")

    # Display sample images
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    fig.suptitle(f"{dataset_name} - Sample Images", fontsize=16)

    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_images[i], cmap="gray")
        ax.set_title(f"Label: {sample_labels[i]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Pixel value information
    image_array = np.array(sample_images)

    print("\nIMAGE PIXEL INFORMATION")
    print("Minimum pixel value:", image_array.min())
    print("Maximum pixel value:", image_array.max())
    print("Data type:", image_array.dtype)


if __name__ == "__main__":

    explore_dataset(
        "MNIST",
        MNIST_PATH,
        "mnist"
    )

    explore_dataset(
        "EMNIST LETTERS",
        EMNIST_PATH,
        "emnist/letters"
    )

    print("\n========== DATASET EXPLORATION COMPLETE ==========")
