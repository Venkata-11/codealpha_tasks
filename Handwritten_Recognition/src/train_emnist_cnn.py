
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATA_PATH = Path("data/preprocessed/emnist_stratified_splits.npz")
MODEL_PATH = Path("models/emnist_cnn.keras")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

print("========== EMNIST CNN TRAINING ==========")

# Load dataset
data = np.load(DATA_PATH)

x_train = data["x_train"]
y_train = data["y_train"]

x_val = data["x_val"]
y_val = data["y_val"]

x_test = data["x_test"]
y_test = data["y_test"]

print("\nDataset loaded successfully!")
print("Training images:", x_train.shape)
print("Validation images:", x_val.shape)
print("Testing images:", x_test.shape)

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(26, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n========== MODEL SUMMARY ==========")
model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_loss",
    save_best_only=True
)

# Train model
print("\n========== STARTING TRAINING ==========")

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, checkpoint]
)

# Save final model with best restored weights
model.save(MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)

# Evaluate once on the separate test set
print("\n========== FINAL TEST EVALUATION ==========")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\n========== TRAINING COMPLETED ==========")
