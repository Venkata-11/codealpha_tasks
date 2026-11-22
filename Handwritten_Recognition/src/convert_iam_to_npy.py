
import numpy as np
import os

SOURCE_PATH = "data/iam/processed"
OUTPUT_PATH = "data/iam/memmap"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("========== IAM MEMORY-MAP CONVERSION ==========")

for split in ["train", "validation", "test"]:

    print(f"\nConverting {split} dataset...")

    source_file = os.path.join(SOURCE_PATH, f"{split}.npz")

    with np.load(source_file) as data:

        for key in [
            "images",
            "labels",
            "image_widths",
            "label_lengths"
        ]:

            array = data[key]

            output_file = os.path.join(
                OUTPUT_PATH,
                f"{split}_{key}.npy"
            )

            np.save(output_file, array)

            print(
                f"Saved {key}: {array.shape}"
            )

            del array

    print(f"{split} conversion completed!")

print("\n========== ALL CONVERSIONS COMPLETED ==========")
