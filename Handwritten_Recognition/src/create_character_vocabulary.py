
import json
import os
from datasets import load_from_disk

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_PATH = "data/iam/huggingface_dataset"
OUTPUT_PATH = "data/iam/character_vocabulary.json"

print("=" * 60)
print("       CRNN CHARACTER VOCABULARY CREATION")
print("=" * 60)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

dataset = load_from_disk(DATASET_PATH)

# Use only training data to construct the vocabulary
train_data = dataset["train"]

# --------------------------------------------------
# COLLECT UNIQUE CHARACTERS
# --------------------------------------------------

characters = set()

for sample in train_data:
    characters.update(sample["text"])

characters = sorted(characters)

print("\nUnique training characters:", len(characters))

# --------------------------------------------------
# CREATE CHARACTER MAPPING
# --------------------------------------------------

# Reserve 0 for CTC blank.
char_to_index = {
    character: index + 1
    for index, character in enumerate(characters)
}

index_to_char = {
    str(index): character
    for character, index in char_to_index.items()
}

blank_index = 0

total_classes = len(characters) + 1

# --------------------------------------------------
# VERIFY ALL DATASET SPLITS
# --------------------------------------------------

print("\n========== VERIFY CHARACTER COVERAGE ==========")

for split_name, split_data in dataset.items():

    missing_characters = set()

    for sample in split_data:
        for character in sample["text"]:

            if character not in char_to_index:
                missing_characters.add(character)

    if missing_characters:
        print(
            f"{split_name}: Missing characters:",
            sorted(missing_characters)
        )
    else:
        print(f"{split_name}: All characters mapped successfully.")

# --------------------------------------------------
# SAVE VOCABULARY
# --------------------------------------------------

vocabulary = {
    "char_to_index": char_to_index,
    "index_to_char": index_to_char,
    "blank_index": blank_index,
    "total_classes": total_classes
}

os.makedirs("data/iam", exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(vocabulary, file, ensure_ascii=False, indent=4)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n========== VOCABULARY SUMMARY ==========")

print("Character classes:", len(characters))
print("CTC blank index:", blank_index)
print("Total output classes:", total_classes)

print("\nFirst 10 character mappings:")

for character, index in list(char_to_index.items())[:10]:

    display_character = repr(character)

    print(f"{display_character} -> {index}")

print("\nVocabulary saved to:", OUTPUT_PATH)

print("\n========== VOCABULARY CREATION COMPLETED ==========")
