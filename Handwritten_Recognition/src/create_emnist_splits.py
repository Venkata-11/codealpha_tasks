
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# File paths
INPUT_PATH = Path("data/preprocessed/emnist_preprocessed.npz")
OUTPUT_PATH = Path("data/preprocessed/emnist_stratified_splits.npz")

print("========== EMNIST STRATIFIED SPLITTING ==========")

# Load preprocessed EMNIST data
data = np.load(INPUT_PATH)

x = data["x_train"]
y = data["y_train"]

print("\nOriginal training dataset:")
print("Images:", x.shape)
print("Labels:", y.shape)

# First split: 80% training, 20% temporary
x_train, x_temp, y_train, y_temp = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: 10% validation, 10% testing
x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Display split information
print("\n========== FINAL DATASET SPLITS ==========")

print("\nTraining:")
print("Images:", x_train.shape)
print("Labels:", y_train.shape)

print("\nValidation:")
print("Images:", x_val.shape)
print("Labels:", y_val.shape)

print("\nTesting:")
print("Images:", x_test.shape)
print("Labels:", y_test.shape)

# Verify that all 26 classes are present
print("\n========== CLASS VERIFICATION ==========")

for name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test)
]:
    classes = np.unique(labels)
    print(f"{name}: {len(classes)} classes")
    assert len(classes) == 26, f"Missing classes in {name}!"

print("\nAll 26 letters are present in all three splits.")

# Save the new dataset
np.savez_compressed(
    OUTPUT_PATH,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test
)

print("\nSaved successfully to:", OUTPUT_PATH)
print("\n========== SPLITTING COMPLETED ==========")
