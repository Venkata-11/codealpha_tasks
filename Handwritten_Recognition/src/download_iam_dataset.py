
import os
from datasets import load_dataset

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_NAME = "Teklia/IAM-line"
SAVE_PATH = "data/iam/huggingface_dataset"

print("=" * 55)
print("       IAM HANDWRITTEN TEXT DATASET DOWNLOAD")
print("=" * 55)

# Create the destination folder
os.makedirs("data/iam", exist_ok=True)

# --------------------------------------------------
# DOWNLOAD DATASET
# --------------------------------------------------

print("\nDownloading IAM dataset from Hugging Face...")
print("This may take some time depending on your internet speed.")

dataset = load_dataset(DATASET_NAME)

# --------------------------------------------------
# DISPLAY DATASET INFORMATION
# --------------------------------------------------

print("\n========== DATASET INFORMATION ==========")

total_samples = 0

for split_name, split_data in dataset.items():

    sample_count = len(split_data)
    total_samples += sample_count

    print(f"\n{split_name.upper()} samples: {sample_count}")
    print("Columns:", split_data.column_names)

print("\nTotal samples:", total_samples)

# --------------------------------------------------
# INSPECT A SAMPLE
# --------------------------------------------------

print("\n========== SAMPLE INSPECTION ==========")

sample = dataset["train"][0]

print("Text transcription:", sample["text"])
print("Image dimensions:", sample["image"].size)
print("Image mode:", sample["image"].mode)

# Save one sample image for inspection
sample["image"].save("data/iam/sample_handwriting.png")

# --------------------------------------------------
# SAVE DATASET LOCALLY
# --------------------------------------------------

print("\nSaving dataset locally...")

dataset.save_to_disk(SAVE_PATH)

print("\nDataset saved successfully!")
print("Location:", SAVE_PATH)

print("\n========== DOWNLOAD COMPLETED ==========")
