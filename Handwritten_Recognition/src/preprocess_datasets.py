
from pathlib import Path
import numpy as np
import tensorflow_datasets as tfds


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MNIST_PATH = PROJECT_ROOT / "data" / "mnist"
EMNIST_PATH = PROJECT_ROOT / "data" / "emnist"

OUTPUT_PATH = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FUNCTION: CONVERT DATASET TO NUMPY ARRAYS
# --------------------------------------------------

def convert_to_numpy(dataset):

    images = []
    labels = []

    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)

    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int64)

    return images, labels


# --------------------------------------------------
# MNIST PREPROCESSING
# --------------------------------------------------

def preprocess_mnist():

    print("\n========== MNIST PREPROCESSING ==========")

    (train_data, test_data), info = tfds.load(
        "mnist",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        data_dir=str(MNIST_PATH)
    )

    # Convert datasets into NumPy arrays
    x_train, y_train = convert_to_numpy(train_data)
    x_test, y_test = convert_to_numpy(test_data)

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape images for CNN input
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Verify labels
    assert len(np.unique(y_train)) == 10

    # Save preprocessed dataset
    np.savez_compressed(
        OUTPUT_PATH / "mnist_preprocessed.npz",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    print("Training images:", x_train.shape)
    print("Testing images:", x_test.shape)

    print("Training labels:", np.unique(y_train))
    print("Testing labels:", np.unique(y_test))

    print("Pixel range:", x_train.min(), "to", x_train.max())
    print("Number of classes:", len(np.unique(y_train)))

    print("MNIST preprocessing completed!")


# --------------------------------------------------
# EMNIST LETTERS PREPROCESSING
# --------------------------------------------------

def preprocess_emnist():

    print("\n========== EMNIST PREPROCESSING ==========")

    (train_data, test_data), info = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        data_dir=str(EMNIST_PATH)
    )

    # Convert datasets into NumPy arrays
    x_train, y_train = convert_to_numpy(train_data)
    x_test, y_test = convert_to_numpy(test_data)

    # Correct EMNIST image orientation
    # by transposing height and width
    x_train = np.transpose(x_train, (0, 2, 1, 3))
    x_test = np.transpose(x_test, (0, 2, 1, 3))

    # Inspect original labels
    print("\nOriginal training label range:",
          y_train.min(), "to", y_train.max())

    print("Original testing label range:",
          y_test.min(), "to", y_test.max())

    print("Unique original training labels:",
          np.unique(y_train))

    print("Unique original testing labels:",
          np.unique(y_test))

    # Convert EMNIST labels from 1-26 to 0-25
    y_train = y_train - 1
    y_test = y_test - 1

    # Verify label conversion
    assert y_train.min() == 0
    assert y_train.max() == 25

    assert y_test.min() >= 0
    assert y_test.max() <= 25

    assert len(np.unique(y_train)) == 26

    print("\nAfter label conversion:")

    print("Training labels:", np.unique(y_train))
    print("Testing labels:", np.unique(y_test))

    print("Number of training classes:",
          len(np.unique(y_train)))

    print("Number of testing classes:",
          len(np.unique(y_test)))

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape images for CNN input
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Save preprocessed dataset
    np.savez_compressed(
        OUTPUT_PATH / "emnist_preprocessed.npz",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    print("\nTraining images:", x_train.shape)
    print("Testing images:", x_test.shape)

    print("Pixel range:", x_train.min(), "to", x_train.max())

    print("Number of training classes:",
          len(np.unique(y_train)))

    print("EMNIST preprocessing completed!")


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

if __name__ == "__main__":

    preprocess_mnist()

    preprocess_emnist()

    print("\n======================================")
    print("ALL PREPROCESSING COMPLETED!")
    print("======================================")

    print("Preprocessed files saved in:")
    print(OUTPUT_PATH)
