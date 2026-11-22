
from collections import Counter
from pathlib import Path
import tensorflow_datasets as tfds


# Project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMNIST_PATH = PROJECT_ROOT / "data" / "emnist"


print("========== VERIFY ORIGINAL EMNIST DATASET ==========")

# Load original EMNIST Letters dataset
(train_data, test_data), info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True,
    data_dir=str(EMNIST_PATH)
)

print("\nDataset name:", info.name)
print("Dataset version:", info.version)

print("\nTraining images:", info.splits["train"].num_examples)
print("Testing images:", info.splits["test"].num_examples)

# Count original training labels
train_counts = Counter(
    int(label)
    for _, label in tfds.as_numpy(train_data)
)

# Count original testing labels
test_counts = Counter(
    int(label)
    for _, label in tfds.as_numpy(test_data)
)

print("\n========== ORIGINAL TRAINING LABELS ==========")

for label, count in sorted(train_counts.items()):
    print(f"Label {label}: {count} images")

print("\nNumber of training classes:", len(train_counts))

print("\n========== ORIGINAL TESTING LABELS ==========")

for label, count in sorted(test_counts.items()):
    print(f"Label {label}: {count} images")

print("\nNumber of testing classes:", len(test_counts))

# Identify missing test labels
all_labels = set(range(1, 27))
missing_labels = sorted(all_labels - set(test_counts.keys()))

print("\nMissing testing labels:", missing_labels)

print("\nOriginal EMNIST verification completed!")
