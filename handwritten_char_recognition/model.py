"""
model.py
CNN architecture for handwritten character/digit recognition.

Two label spaces are supported:
  - "digits":  10 classes   (MNIST)              -> 0-9
  - "chars":   47 classes   (EMNIST 'balanced')   -> 0-9, A-Z, a-z (merged case-ambiguous letters)

The same CNN backbone is reused for both; only the final Dense layer size changes.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 28  # EMNIST/MNIST are 28x28 grayscale


def build_cnn(num_classes: int, img_size: int = IMG_SIZE) -> tf.keras.Model:
    """Build a compact but accurate CNN for 28x28 grayscale character images.

    Architecture: 3 conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool) with
    increasing filter counts, a dropout-regularized dense head, and a
    softmax output. This is small enough to train on CPU in a few minutes
    per epoch on MNIST/EMNIST, but deep enough to comfortably exceed 99%
    on MNIST and ~90% on EMNIST-balanced.
    """
    inputs = layers.Input(shape=(img_size, img_size, 1))

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"char_cnn_{num_classes}cls")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# EMNIST 'balanced' split label map (47 classes) -> printable character.
# Order matches the official EMNIST mapping file.
EMNIST_BALANCED_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
# Note: EMNIST-balanced merges visually-similar upper/lower pairs
# (C,I,J,K,L,M,O,P,S,U,V,W,X,Y,Z) into a single class using the uppercase glyph,
# so only the lowercase letters that look distinct from their uppercase
# form (a,b,d,e,f,g,h,n,q,r,t) get their own class -> 10 + 26 + 11 = 47.

DIGIT_LABELS = list("0123456789")
