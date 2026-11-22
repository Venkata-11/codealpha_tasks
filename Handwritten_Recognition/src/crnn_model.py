
import tensorflow as tf
from tensorflow.keras import layers, Model


# ==========================================
# CRNN CONFIGURATION
# ==========================================

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 2560
OUTPUT_CLASSES = 80

print("========== BUILDING CRNN MODEL ==========")


# ==========================================
# INPUT
# ==========================================

inputs = layers.Input(
    shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1),
    name="image"
)


# ==========================================
# CNN FEATURE EXTRACTION
# ==========================================

x = layers.Conv2D(
    32, (3, 3), padding="same", activation="relu"
)(inputs)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    pool_size=(2, 2), padding="same"
)(x)


x = layers.Conv2D(
    64, (3, 3), padding="same", activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    pool_size=(2, 2), padding="same"
)(x)


x = layers.Conv2D(
    128, (3, 3), padding="same", activation="relu"
)(x)

x = layers.BatchNormalization()(x)


# ==========================================
# CONVERT CNN FEATURES INTO SEQUENCE
# ==========================================

# CNN output: (batch, 16, 640, 128)
# Rearrange to: (batch, 640, 16, 128)

x = layers.Permute((2, 1, 3))(x)

# Combine height and channel dimensions.
# Result: (batch, 640, 2048)

x = layers.Reshape((640, 16 * 128))(x)

# Reduce feature dimension.

x = layers.Dense(
    256,
    activation="relu"
)(x)


# ==========================================
# BILSTM SEQUENCE LEARNING
# ==========================================

x = layers.Bidirectional(
    layers.LSTM(128, return_sequences=True)
)(x)

x = layers.Dropout(0.25)(x)


# ==========================================
# CHARACTER PREDICTION
# ==========================================

# No softmax here.
# CTC loss receives raw logits.

outputs = layers.Dense(
    OUTPUT_CLASSES,
    name="character_logits"
)(x)


# ==========================================
# CREATE MODEL
# ==========================================

crnn_model = Model(
    inputs=inputs,
    outputs=outputs,
    name="IAM_CRNN"
)


# ==========================================
# MODEL BUILDER FUNCTION
# ==========================================

def build_crnn_model():
    return crnn_model


# ==========================================
# DISPLAY MODEL SUMMARY
# ==========================================

print("\nCRNN architecture created successfully!")

crnn_model.summary()


# ==========================================
# VERIFY OUTPUT SHAPE
# ==========================================

dummy_input = tf.zeros((1, 64, 2560, 1))

dummy_output = crnn_model(dummy_input)

print("\n========== MODEL OUTPUT CHECK ==========")

print("Input shape:", dummy_input.shape)
print("Output shape:", dummy_output.shape)

print("\nExpected output:")
print("(1, 640, 80)")

assert dummy_output.shape == (1, 640, 80), (
    "CRNN output shape does not match expected shape!"
)

print("\nCRNN model verification successful!")
