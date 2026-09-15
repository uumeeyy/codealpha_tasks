"""
crnn_extension.py
Skeleton for extending single-character recognition to full word/sentence
recognition, as sketched in the task brief ("Extendable to full word or
sentence recognition with sequence modeling (like CRNN)").

Why a CRNN instead of the CNN classifier in model.py:
  The CNN in model.py assumes one pre-segmented character per image and
  outputs a single softmax over a fixed label set. A handwritten word is a
  variable-length sequence of characters that are hard to segment reliably
  (cursive joins, uneven spacing). A CRNN sidesteps segmentation entirely:

  1. CNN backbone   -> extracts a feature map from the full word image.
  2. Reshape        -> the feature map is sliced into a left-to-right
                        sequence of "time steps" (one per horizontal
                        position), each a feature vector.
  3. BiLSTM layers   -> model context along that sequence in both directions
                        (e.g. an 'r' looks different next to 'r' vs 'n').
  4. Dense + CTC     -> a softmax over (alphabet + blank) per time step,
                        trained with Connectionist Temporal Classification
                        (CTC) loss, which lets the network learn its own
                        alignment between image columns and output
                        characters -- no manual segmentation needed.

Typical datasets for training this stage: IAM Handwriting Database or
custom EMNIST-composited words (concatenate random EMNIST characters
into synthetic "words" with random spacing/rotation as a bootstrap
dataset before fine-tuning on real handwriting like IAM).

This file defines the architecture and a CTC loss wrapper; it is not
wired to a dataset loader, since that depends on which corpus you pick.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_crnn(img_height: int = 32, img_width: int = 128, num_classes: int = 80) -> tf.keras.Model:
    """num_classes should be (size of your character alphabet) + 1 for the CTC blank token."""
    inputs = layers.Input(shape=(img_height, img_width, 1), name="image")
    labels = layers.Input(shape=(None,), dtype="int32", name="label")
    input_length = layers.Input(shape=(1,), dtype="int32", name="input_length")
    label_length = layers.Input(shape=(1,), dtype="int32", name="label_length")

    # --- CNN feature extractor ---
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)  # /2

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)  # /4

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 1))(x)  # collapse height only -> /8 height, /4 width

    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Collapse the height dimension so each remaining horizontal position
    # becomes one "time step" in the sequence fed to the RNN.
    new_h = x.shape[1]
    new_w = x.shape[2]
    new_c = x.shape[3]
    x = layers.Reshape((new_w, new_h * new_c))(x)  # (batch, time_steps, features)
    x = layers.Dense(128, activation="relu")(x)

    # --- Sequence modeling ---
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)

    y_pred = layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)

    ctc_loss = layers.Lambda(_ctc_loss_fn, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    training_model = models.Model(
        inputs=[inputs, labels, input_length, label_length], outputs=ctc_loss
    )
    # Identity loss: the Lambda layer already computes per-sample CTC loss.
    training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss={"ctc": lambda y_true, y_pred: y_pred})

    inference_model = models.Model(inputs=inputs, outputs=y_pred)

    return training_model, inference_model


def _ctc_loss_fn(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_greedy_decode(y_pred, input_length, alphabet: list[str]) -> list[str]:
    """Decode CRNN softmax output into strings using greedy CTC decoding."""
    decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=input_length, greedy=True)
    sequences = decoded[0].numpy()
    results = []
    for seq in sequences:
        chars = [alphabet[i] for i in seq if i >= 0]
        results.append("".join(chars))
    return results


if __name__ == "__main__":
    # Sanity check: build the graph and print a summary of the inference model.
    _, inference_model = build_crnn()
    inference_model.summary()
