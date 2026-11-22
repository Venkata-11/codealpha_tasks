
import numpy as np
import json
import os

DATA_PATH = "data/iam/processed"

with open(
    "data/iam/character_vocabulary.json",
    "r",
    encoding="utf-8"
) as file:
    vocabulary = json.load(file)

print("========== CTC SEQUENCE LENGTH CHECK ==========")

for split in ["train", "validation", "test"]:

    file_path = os.path.join(DATA_PATH, f"{split}.npz")

    with np.load(file_path) as data:
        widths = data["image_widths"]
        labels = data["labels"]
        label_lengths = data["label_lengths"]

        # CNN reduces width by a factor of 8.
        output_lengths = (widths + 3) // 4

        required_lengths = []

        for i, length in enumerate(label_lengths):
            label = labels[i, :length]

            # Consecutive repeated characters need
            # an additional CTC time step.
            repeated = np.sum(label[1:] == label[:-1])

            required_lengths.append(length + repeated)

        required_lengths = np.array(required_lengths)

        valid = output_lengths >= required_lengths

        print(f"\n{split.upper()}")
        print("Total samples:", len(widths))
        print("CTC-compatible samples:", np.sum(valid))
        print("Incompatible samples:", np.sum(~valid))
        print(
            "Compatibility:",
            round(np.mean(valid) * 100, 2),
            "%"
        )

        if np.any(~valid):
            print("First incompatible sample indices:",
                  np.where(~valid)[0][:10])

print("\n========== CHECK COMPLETED ==========")
