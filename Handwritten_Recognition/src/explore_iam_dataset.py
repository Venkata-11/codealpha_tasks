
from datasets import load_from_disk
import numpy as np

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

DATASET_PATH = "data/iam/huggingface_dataset"

print("=" * 60)
print("       IAM HANDWRITTEN DATASET EXPLORATION")
print("=" * 60)

dataset = load_from_disk(DATASET_PATH)

# --------------------------------------------------
# EXPLORE DATASET SPLITS
# --------------------------------------------------

print("\n========== DATASET SPLITS ==========")

for split_name, split_data in dataset.items():

    print(f"{split_name.upper()} samples: {len(split_data)}")
    print("Columns:", split_data.column_names)

# --------------------------------------------------
# EXPLORE IMAGE DIMENSIONS
# --------------------------------------------------

print("\n========== IMAGE DIMENSIONS ==========")

train_data = dataset["train"]

widths = []
heights = []

for sample in train_data:
    image = sample["image"]
    width, height = image.size

    widths.append(width)
    heights.append(height)

print("Minimum width:", min(widths))
print("Maximum width:", max(widths))
print("Average width:", round(np.mean(widths), 2))

print("Minimum height:", min(heights))
print("Maximum height:", max(heights))
print("Average height:", round(np.mean(heights), 2))

# --------------------------------------------------
# EXPLORE TEXT LABELS
# --------------------------------------------------

print("\n========== TEXT LABEL ANALYSIS ==========")

text_labels = [sample["text"] for sample in train_data]

text_lengths = [len(text) for text in text_labels]

all_characters = set()

for text in text_labels:
    all_characters.update(text)

print("Minimum text length:", min(text_lengths))
print("Maximum text length:", max(text_lengths))
print("Average text length:", round(np.mean(text_lengths), 2))

print("Number of unique characters:", len(all_characters))
print("Characters:", "".join(sorted(all_characters)))

# --------------------------------------------------
# DISPLAY SAMPLE IMAGE AND TEXT PAIRS
# --------------------------------------------------

print("\n========== SAMPLE TRANSCRIPTIONS ==========")

for index in range(5):

    sample = train_data[index]

    print(f"\nSample {index + 1}")
    print("Transcription:", sample["text"])
    print("Image size:", sample["image"].size)

# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\n========== EXPLORATION COMPLETED ==========")
