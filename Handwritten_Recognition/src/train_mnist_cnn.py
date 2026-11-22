
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# File paths
DATA_PATH = Path("data/preprocessed/mnist_preprocessed.npz")
MODEL_PATH = Path("models/mnist_cnn.keras")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

print("========== MNIST CNN TRAINING ==========")

# Load preprocessed MNIST dataset
data = np.load(DATA_PATH)

x = data["x_train"]
y = data["y_train"]

x_test = data["x_test"]
y_test = data["y_test"]

print("\nDataset loaded successfully!")
print("Original training images:", x.shape)
print("Testing images:", x_test.shape)

# Create stratified training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

print("\n========== DATASET SPLITS ==========")
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

    layers.Dense(10, activation="softmax")
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

# Save model with the best validation weights restored
model.save(MODEL_PATH)

print("\nBest model saved to:", MODEL_PATH)

# Evaluate on untouched MNIST test dataset
print("\n========== FINAL TEST EVALUATION ==========")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\n========== MNIST TRAINING COMPLETED ==========")
