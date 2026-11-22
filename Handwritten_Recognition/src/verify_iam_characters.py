
from datasets import load_from_disk

# Load the downloaded dataset
dataset = load_from_disk("data/iam/huggingface_dataset")

print("=" * 60)
print("       IAM CHARACTER VOCABULARY VERIFICATION")
print("=" * 60)

# Collect characters from each split
split_characters = {}

for split_name, split_data in dataset.items():

    characters = set()

    for sample in split_data:
        characters.update(sample["text"])

    split_characters[split_name] = characters

    print(f"\n{split_name.upper()}")
    print("Unique characters:", len(characters))
    print("Characters:", "".join(sorted(characters)))

# Compare character coverage
train_chars = split_characters["train"]

print("\n========== CHARACTER COVERAGE ==========")

for split_name in ["validation", "test"]:

    missing = split_characters[split_name] - train_chars

    print(f"\n{split_name.upper()}")

    if missing:
        print("Characters missing from training:", sorted(missing))
    else:
        print("All characters are present in training data.")

print("\n========== VERIFICATION COMPLETED ==========")
