"""
train.py
Train the CNN on MNIST (digits only) or EMNIST-balanced (digits + letters).

Usage:
    python train.py --dataset mnist   --epochs 15
    python train.py --dataset emnist  --epochs 15

Outputs:
    models/<dataset>_cnn.keras   -- trained model
    models/<dataset>_history.png -- accuracy/loss curves
    A printed classification report + confusion matrix summary on the test set.

Data source:
    - mnist:  tf.keras.datasets.mnist (auto-downloaded by Keras)
    - emnist: tensorflow-datasets ("emnist/balanced") -- first run downloads
              ~536MB and caches it under ~/tensorflow_datasets.
              If you don't have internet access in your runtime, download
              the EMNIST dataset manually from
              https://www.nist.gov/itl/products-and-services/emnist-dataset
              and adapt `load_emnist()` to read the .mat/.gz files instead.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import build_cnn, IMG_SIZE, EMNIST_BALANCED_LABELS, DIGIT_LABELS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test), DIGIT_LABELS


def load_emnist():
    """Loads EMNIST-balanced via tensorflow-datasets.

    EMNIST images are stored transposed relative to MNIST convention, so we
    correct orientation to match standard upright characters.
    """
    import tensorflow_datasets as tfds

    ds_train, ds_test = tfds.load(
        "emnist/balanced", split=["train", "test"], as_supervised=True
    )

    def to_numpy(ds):
        images, labels = [], []
        for img, label in tfds.as_numpy(ds):
            images.append(img)
            labels.append(label)
        x = np.array(images).squeeze(-1)
        # EMNIST is stored rotated 90 deg + flipped vs. natural reading orientation
        x = np.transpose(x, (0, 2, 1))
        y = np.array(labels)
        return x, y

    x_train, y_train = to_numpy(ds_train)
    x_test, y_test = to_numpy(ds_test)
    return (x_train, y_train), (x_test, y_test), EMNIST_BALANCED_LABELS


def preprocess(x, y):
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, -1)  # (N, 28, 28, 1)
    return x, y


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    if args.dataset == "mnist":
        (x_train, y_train), (x_test, y_test), labels = load_mnist()
    else:
        (x_train, y_train), (x_test, y_test), labels = load_emnist()

    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}, classes: {len(labels)}")

    # Light augmentation helps generalize to messy real-world handwriting
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomZoom(0.08),
    ])

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10_000)
        .batch(args.batch_size)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_cnn(num_classes=len(labels))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Final test accuracy: {test_acc:.4f}")

    model_path = os.path.join(MODELS_DIR, f"{args.dataset}_cnn.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    plot_history(history, os.path.join(MODELS_DIR, f"{args.dataset}_history.png"))

    with open(os.path.join(MODELS_DIR, f"{args.dataset}_labels.txt"), "w") as f:
        f.write("\n".join(labels))


if __name__ == "__main__":
    main()
