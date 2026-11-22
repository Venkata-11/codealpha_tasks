
import os
import time
import numpy as np
import tensorflow as tf

from crnn_model import build_crnn_model
from iam_data_loader import IAMDataLoader


# ==========================================
# CONFIGURATION
# ==========================================

BATCH_SIZE = 2
LEARNING_RATE = 0.0001

# Benchmark limits.
# We will increase these for full training later.

MAX_TRAIN_BATCHES = 10
MAX_VALIDATION_BATCHES = 3

MODEL_DIRECTORY = "models"
CHECKPOINT_PATH = os.path.join(
    MODEL_DIRECTORY,
    "iam_crnn_benchmark.keras"
)

os.makedirs(MODEL_DIRECTORY, exist_ok=True)


# ==========================================
# CTC LOSS FUNCTION
# ==========================================

def calculate_ctc_loss(model, batch, training):

    images = tf.convert_to_tensor(
        batch["images"],
        dtype=tf.float32
    )

    labels = tf.convert_to_tensor(
        batch["labels"],
        dtype=tf.int32
    )

    input_lengths = tf.convert_to_tensor(
        batch["input_lengths"],
        dtype=tf.int32
    )

    label_lengths = tf.convert_to_tensor(
        batch["label_lengths"],
        dtype=tf.int32
    )

    logits = model(images, training=training)

    loss_values = tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_lengths,
        logit_length=input_lengths,
        logits_time_major=False,
        blank_index=0
    )

    return tf.reduce_mean(loss_values)


# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader, optimizer):

    total_loss = 0.0
    batch_count = 0

    start_time = time.perf_counter()

    for batch_number, batch in enumerate(loader, start=1):

        with tf.GradientTape() as tape:

            loss = calculate_ctc_loss(
                model,
                batch,
                training=True
            )

        if not np.isfinite(loss.numpy()):
            raise RuntimeError(
                f"Non-finite training loss at batch {batch_number}"
            )

        gradients = tape.gradient(
            loss,
            model.trainable_variables
        )

        valid_gradients = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                model.trainable_variables
            )
            if gradient is not None
        ]

        if not valid_gradients:
            raise RuntimeError("No gradients generated!")

        optimizer.apply_gradients(valid_gradients)

        total_loss += float(loss.numpy())
        batch_count += 1

        if batch_number % 2 == 0:
            print(
                f"Training batch {batch_number}/"
                f"{MAX_TRAIN_BATCHES}: "
                f"Loss = {float(loss.numpy()):.4f}"
            )

        if batch_number >= MAX_TRAIN_BATCHES:
            break

    elapsed_time = time.perf_counter() - start_time

    average_loss = total_loss / batch_count

    average_batch_time = elapsed_time / batch_count

    print("\nTraining summary:")
    print("Batches processed:", batch_count)
    print("Average training loss:", round(average_loss, 4))
    print("Total training time:", round(elapsed_time, 2), "seconds")
    print("Average time per batch:", round(average_batch_time, 2), "seconds")

    return average_loss, average_batch_time


# ==========================================
# VALIDATION FUNCTION
# ==========================================

def validate_model(model, loader):

    total_loss = 0.0
    batch_count = 0

    print("\nStarting validation...")

    for batch_number, batch in enumerate(loader, start=1):

        loss = calculate_ctc_loss(
            model,
            batch,
            training=False
        )

        if not np.isfinite(loss.numpy()):
            raise RuntimeError(
                f"Non-finite validation loss at batch {batch_number}"
            )

        total_loss += float(loss.numpy())
        batch_count += 1

        print(
            f"Validation batch {batch_number}/"
            f"{MAX_VALIDATION_BATCHES}: "
            f"Loss = {float(loss.numpy()):.4f}"
        )

        if batch_number >= MAX_VALIDATION_BATCHES:
            break

    average_loss = total_loss / batch_count

    print("\nValidation loss:", round(average_loss, 4))

    return average_loss


# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("             CRNN TRAINING")
    print("=" * 55)

    # Load model.

    print("\nSTEP 1: Loading CRNN model...")

    model = build_crnn_model()

    print("Model loaded successfully!")
    print("Parameters:", model.count_params())

    # Load training data.

    print("\nSTEP 2: Loading training dataset...")

    train_loader = IAMDataLoader(
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Load validation data.

    print("\nSTEP 3: Loading validation dataset...")

    validation_loader = IAMDataLoader(
        split="validation",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Initialize optimizer.

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=5.0
    )

    print("\nSTEP 4: Starting benchmark training...")

    training_loss, batch_time = train_one_epoch(
        model,
        train_loader,
        optimizer
    )

    validation_loss = validate_model(
        model,
        validation_loader
    )

    # Save benchmark model.

    print("\nSTEP 5: Saving benchmark checkpoint...")

    model.save(CHECKPOINT_PATH)

    print("Benchmark model saved:", CHECKPOINT_PATH)

    # Final report.

    print("\n" + "=" * 55)
    print("         CRNN BENCHMARK COMPLETED")
    print("=" * 55)

    print("Training loss:", round(training_loss, 4))
    print("Validation loss:", round(validation_loss, 4))
    print("Average batch time:", round(batch_time, 2), "seconds")

    print("\nAll benchmark operations completed successfully!")
