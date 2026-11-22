
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

file_path = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "emnist_preprocessed.npz"
)

data = np.load(file_path)

y_train = data["y_train"]
y_test = data["y_test"]

print("========== SAVED EMNIST LABEL CHECK ==========")

print("\nTraining label range:",
      y_train.min(), "to", y_train.max())

print("Testing label range:",
      y_test.min(), "to", y_test.max())

print("\nTraining label counts:")
print(dict(zip(*np.unique(y_train, return_counts=True))))

print("\nTesting label counts:")
print(dict(zip(*np.unique(y_test, return_counts=True))))

print("\nNumber of training classes:", len(np.unique(y_train)))
print("Number of testing classes:", len(np.unique(y_test)))
