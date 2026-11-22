
import tensorflow_datasets as tfds

# Project dataset locations
MNIST_PATH = "data/mnist"
EMNIST_PATH = "data/emnist"


def download_mnist():
    print("\n========== DOWNLOADING MNIST ==========")

    (train_data, test_data), info = tfds.load(
        "mnist",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        data_dir=MNIST_PATH
    )

    print("MNIST downloaded successfully!")
    print("Training images:", info.splits["train"].num_examples)
    print("Testing images:", info.splits["test"].num_examples)
    print("Image shape:", info.features["image"].shape)
    print("Number of classes:", info.features["label"].num_classes)

    return train_data, test_data


def download_emnist():
    print("\n========== DOWNLOADING EMNIST LETTERS ==========")

    (train_data, test_data), info = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        data_dir=EMNIST_PATH
    )

    print("EMNIST Letters downloaded successfully!")
    print("Training images:", info.splits["train"].num_examples)
    print("Testing images:", info.splits["test"].num_examples)
    print("Image shape:", info.features["image"].shape)
    print("Number of classes:", info.features["label"].num_classes)

    return train_data, test_data


if __name__ == "__main__":
    mnist_train, mnist_test = download_mnist()
    emnist_train, emnist_test = download_emnist()

    print("\n========== DATASET COLLECTION COMPLETE ==========")
