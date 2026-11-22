
from datasets import load_from_disk
from PIL import Image
import numpy as np
import json
import os

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = "data/iam/huggingface_dataset"
VOCAB_PATH = "data/iam/character_vocabulary.json"
OUTPUT_PATH = "data/iam/processed"

TARGET_HEIGHT = 64
MAX_WIDTH = 2560

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("========== IAM DATA PREPROCESSING ==========")

# ==========================================
# LOAD DATASET AND VOCABULARY
# ==========================================

dataset = load_from_disk(DATASET_PATH)

with open(VOCAB_PATH, "r", encoding="utf-8") as file:
    vocabulary = json.load(file)

char_to_index = vocabulary["char_to_index"]

# ==========================================
# PREPROCESS ONE IMAGE
# ==========================================

def preprocess_image(image):
    image = image.convert("L")

    original_width, original_height = image.size

    # Preserve aspect ratio
    new_width = round(
        original_width * TARGET_HEIGHT / original_height
    )

    # Resize image
    image = image.resize(
        (new_width, TARGET_HEIGHT),
        Image.Resampling.LANCZOS
    )

    # Convert to NumPy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0

    # White padding on the right
    padded_image = np.ones(
        (TARGET_HEIGHT, MAX_WIDTH),
        dtype=np.float32
    )

    padded_image[:, :new_width] = image_array

    return padded_image, new_width


# ==========================================
# ENCODE TRANSCRIPTION
# ==========================================

def encode_text(text):
    return [char_to_index[char] for char in text]


# ==========================================
# PROCESS EACH DATASET SPLIT
# ==========================================

for split_name in ["train", "validation", "test"]:

    print(f"\nProcessing {split_name} dataset...")

    images = []
    labels = []
    image_widths = []
    label_lengths = []

    for sample in dataset[split_name]:

        # Process image
        processed_image, actual_width = preprocess_image(
            sample["image"]
        )

        # Encode transcription
        encoded_label = encode_text(sample["text"])

        images.append(processed_image)
        labels.append(encoded_label)
        image_widths.append(actual_width)
        label_lengths.append(len(encoded_label))

    # Convert to NumPy arrays
    images = np.array(images, dtype=np.float32)
    image_widths = np.array(image_widths, dtype=np.int32)
    label_lengths = np.array(label_lengths, dtype=np.int32)

    # Labels have different lengths.
    # Pad labels with zero (CTC blank index).
    max_label_length = max(label_lengths)

    padded_labels = np.zeros(
        (len(labels), max_label_length),
        dtype=np.int32
    )

    for index, label in enumerate(labels):
        padded_labels[index, :len(label)] = label

    # Save processed data
    np.savez_compressed(
        os.path.join(OUTPUT_PATH, f"{split_name}.npz"),
        images=images,
        labels=padded_labels,
        image_widths=image_widths,
        label_lengths=label_lengths
    )

    print("Images shape:", images.shape)
    print("Labels shape:", padded_labels.shape)
    print("Image widths shape:", image_widths.shape)
    print("Label lengths shape:", label_lengths.shape)

    print(
        "Saved:",
        os.path.join(OUTPUT_PATH, f"{split_name}.npz")
    )

print("\n========== PREPROCESSING COMPLETED ==========")
