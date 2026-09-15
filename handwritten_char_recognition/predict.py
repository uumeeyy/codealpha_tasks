"""
predict.py
Run inference on a single image file (e.g. a photo/scan of a handwritten
character, or a canvas drawing exported as PNG).

Usage:
    python predict.py --image path/to/char.png --dataset mnist
    python predict.py --image path/to/char.png --dataset emnist --topk 3
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from model import IMG_SIZE


def load_labels(dataset: str):
    labels_path = os.path.join(os.path.dirname(__file__), "models", f"{dataset}_labels.txt")
    with open(labels_path) as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_image(path: str) -> np.ndarray:
    """Convert an arbitrary input image into MNIST/EMNIST-style 28x28 input:
    grayscale, white digit on black background, centered on its bounding box.
    """
    img = Image.open(path).convert("L")

    # Heuristic: if the image looks like dark strokes on a light background
    # (typical for a photo of pen-on-paper or a default white canvas),
    # invert it so the character is bright and the background is dark,
    # matching MNIST/EMNIST convention.
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)
        arr = np.array(img)

    # Crop to the bounding box of the stroke to center the character,
    # the same way MNIST/EMNIST characters are pre-centered.
    ys, xs = np.where(arr > 30)
    if len(xs) > 0 and len(ys) > 0:
        pad = 4
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, arr.shape[1])
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, arr.shape[0])
        img = img.crop((x0, y0, x1, y1))

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), "models", f"{args.dataset}_cnn.keras")
    model = tf.keras.models.load_model(model_path)
    labels = load_labels(args.dataset)

    x = preprocess_image(args.image)
    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][: args.topk]

    for rank, idx in enumerate(top_idx, start=1):
        print(f"{rank}. '{labels[idx]}'  ({probs[idx] * 100:.2f}%)")


if __name__ == "__main__":
    main()
