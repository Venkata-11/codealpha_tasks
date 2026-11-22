
import numpy as np
import os

DATA_PATH = "data/iam/processed"

print("========== VERIFY PROCESSED IAM DATA ==========")

for split in ["train", "validation", "test"]:

    file_path = os.path.join(DATA_PATH, f"{split}.npz")

    print(f"\nChecking {split.upper()}...")

    with np.load(file_path) as data:

        images = data["images"]
        labels = data["labels"]
        image_widths = data["image_widths"]
        label_lengths = data["label_lengths"]

        print("Images:", images.shape)
        print("Labels:", labels.shape)
        print("Image widths:", image_widths.shape)
        print("Label lengths:", label_lengths.shape)

        print("Image minimum:", images.min())
        print("Image maximum:", images.max())

        print("All images finite:", np.isfinite(images).all())
        print("All widths valid:", np.all(
            (image_widths > 0) & (image_widths <= 2560)
        ))
        print("All label lengths valid:", np.all(
            (label_lengths > 0) & (label_lengths <= labels.shape[1])
        ))

        print("Total images:", len(images))

print("\n========== VERIFICATION COMPLETED ==========")
