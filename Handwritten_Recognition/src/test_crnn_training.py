
import numpy as np
import tensorflow as tf

from crnn_model import build_crnn_model
from iam_data_loader import IAMDataLoader


# ==========================================
# CONFIGURATION
# ==========================================

BATCH_SIZE = 2
LEARNING_RATE = 0.0001

print("=" * 55)
print("          CRNN TRAINING PIPELINE TEST")
print("=" * 55)

print("\nTensorFlow version:", tf.__version__)


# ==========================================
# STEP 1: LOAD MODEL
# ==========================================

print("\nSTEP 1: Loading CRNN model...")

model = build_crnn_model()

print("CRNN model loaded successfully!")
print("Total parameters:", model.count_params())


# ==========================================
# STEP 2: LOAD TRAINING DATA
# ==========================================

print("\nSTEP 2: Loading one training batch...")

train_loader = IAMDataLoader(
    split="train",
    batch_size=BATCH_SIZE,
    shuffle=True
)

batch = next(iter(train_loader))

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

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Input lengths:", input_lengths.numpy())
print("Label lengths:", label_lengths.numpy())


# ==========================================
# STEP 3: INITIALIZE OPTIMIZER
# ==========================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


# ==========================================
# STEP 4: FORWARD PASS AND CTC LOSS
# ==========================================

print("\nSTEP 3: Calculating predictions and CTC loss...")

with tf.GradientTape() as tape:

    logits = model(images, training=True)

    print("CRNN output shape:", logits.shape)

    loss_values = tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_lengths,
        logit_length=input_lengths,
        logits_time_major=False,
        blank_index=0
    )

    loss = tf.reduce_mean(loss_values)

print("Individual sample losses:", loss_values.numpy())
print("Average CTC loss:", float(loss.numpy()))


# ==========================================
# STEP 5: CALCULATE GRADIENTS
# ==========================================

print("\nSTEP 4: Calculating gradients...")

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
    raise RuntimeError("No gradients were generated!")

gradient_norm = tf.linalg.global_norm(
    [gradient for gradient, _ in valid_gradients]
)

print("Variables with gradients:", len(valid_gradients))
print("Total trainable variables:", len(model.trainable_variables))
print("Gradient norm:", float(gradient_norm.numpy()))


# ==========================================
# STEP 6: UPDATE MODEL WEIGHTS
# ==========================================

print("\nSTEP 5: Updating model weights...")

optimizer.apply_gradients(valid_gradients)

print("Model weights updated successfully!")


# ==========================================
# STEP 7: VERIFY TRAINING
# ==========================================

if not np.isfinite(loss.numpy()):
    raise RuntimeError("CTC loss is not finite!")

if not np.isfinite(gradient_norm.numpy()):
    raise RuntimeError("Gradient norm is not finite!")

print("\n" + "=" * 55)
print("       CRNN TRAINING TEST SUCCESSFUL!")
print("=" * 55)

print("Forward pass: SUCCESS")
print("CTC loss: SUCCESS")
print("Gradient calculation: SUCCESS")
print("Weight update: SUCCESS")

print("\nYour CRNN is ready for the next training stage!")
