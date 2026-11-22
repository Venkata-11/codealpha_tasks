
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "data/iam/memmap"

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 2560
BATCH_SIZE = 2

print("========== IAM DATA LOADER ==========")


# ==========================================
# DATA LOADER CLASS
# ==========================================

class IAMDataLoader:

    def __init__(self, split, batch_size=BATCH_SIZE, shuffle=False):

        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load arrays using memory mapping.
        # Only the required batches will be read.

        self.images = np.load(
            os.path.join(DATA_PATH, f"{split}_images.npy"),
            mmap_mode="r"
        )

        self.labels = np.load(
            os.path.join(DATA_PATH, f"{split}_labels.npy"),
            mmap_mode="r"
        )

        self.image_widths = np.load(
            os.path.join(DATA_PATH, f"{split}_image_widths.npy"),
            mmap_mode="r"
        )

        self.label_lengths = np.load(
            os.path.join(DATA_PATH, f"{split}_label_lengths.npy"),
            mmap_mode="r"
        )

        self.num_samples = len(self.images)

        print(f"\n{split.upper()} loader initialized")
        print("Total samples:", self.num_samples)
        print("Batch size:", self.batch_size)
        print("Number of batches:", len(self))

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):

        indices = np.arange(self.num_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):

            batch_indices = indices[
                start:start + self.batch_size
            ]

            # Read only the current batch.
            images = np.asarray(
                self.images[batch_indices],
                dtype=np.float32
            )

            labels = np.asarray(
                self.labels[batch_indices],
                dtype=np.int32
            )

            widths = np.asarray(
                self.image_widths[batch_indices],
                dtype=np.int32
            )

            label_lengths = np.asarray(
                self.label_lengths[batch_indices],
                dtype=np.int32
            )

            # Add channel dimension for CNN.
            images = np.expand_dims(images, axis=-1)

            # CNN uses two width-reducing pooling layers.
            # Total width reduction = 4.
            input_lengths = (widths + 3) // 4

            yield {
                "images": images,
                "labels": labels,
                "input_lengths": input_lengths,
                "label_lengths": label_lengths
            }


# ==========================================
# TEST DATA LOADER
# ==========================================

if __name__ == "__main__":

    train_loader = IAMDataLoader(
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("\n========== TESTING FIRST BATCH ==========")

    first_batch = next(iter(train_loader))

    print("Images shape:", first_batch["images"].shape)
    print("Labels shape:", first_batch["labels"].shape)

    print(
        "Input lengths:",
        first_batch["input_lengths"]
    )

    print(
        "Label lengths:",
        first_batch["label_lengths"]
    )

    print(
        "Images normalized:",
        first_batch["images"].min() >= 0
        and first_batch["images"].max() <= 1
    )

    print("\nData loader test completed!")
