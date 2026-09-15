"""
app.py
Interactive Streamlit demo: draw a character on a canvas and the CNN
predicts it live, with a bar chart of the top-5 class probabilities.

Run:
    streamlit run app.py

Requires a trained model at models/mnist_cnn.keras and/or
models/emnist_cnn.keras (see train.py) plus their *_labels.txt files.
"""

import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from model import IMG_SIZE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Handwritten Character Recognition", page_icon="✍️", layout="centered")


@st.cache_resource
def load_model_and_labels(dataset: str):
    model_path = os.path.join(MODELS_DIR, f"{dataset}_cnn.keras")
    labels_path = os.path.join(MODELS_DIR, f"{dataset}_labels.txt")
    if not (os.path.exists(model_path) and os.path.exists(labels_path)):
        return None, None
    model = tf.keras.models.load_model(model_path)
    with open(labels_path) as f:
        labels = [line.strip() for line in f if line.strip()]
    return model, labels


def canvas_to_model_input(canvas_image: np.ndarray) -> np.ndarray | None:
    """RGBA canvas array (white strokes on transparent bg) -> 28x28 model input."""
    if canvas_image is None:
        return None
    alpha = canvas_image[:, :, 3]
    if alpha.max() == 0:
        return None  # nothing drawn yet

    ys, xs = np.where(alpha > 30)
    pad = 15
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, alpha.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, alpha.shape[0])

    cropped = alpha[y0:y1, x0:x1]
    img = Image.fromarray(cropped).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)


st.title("✍️ Handwritten Character Recognition")
st.caption("CNN trained on MNIST (digits) / EMNIST-balanced (digits + letters)")

dataset = st.radio("Recognition mode", ["mnist", "emnist"], format_func=lambda d: "Digits only (MNIST)" if d == "mnist" else "Digits + Letters (EMNIST)", horizontal=True)

model, labels = load_model_and_labels(dataset)

if model is None:
    st.error(
        f"No trained model found for '{dataset}'. Run:\n\n"
        f"    python train.py --dataset {dataset}\n\n"
        f"first, then relaunch this app."
    )
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw here")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    if canvas_result.image_data is not None:
        x = canvas_to_model_input(canvas_result.image_data)
        if x is not None:
            probs = model.predict(x, verbose=0)[0]
            top5 = np.argsort(probs)[::-1][:5]

            best_idx = top5[0]
            st.metric("Predicted character", f"'{labels[best_idx]}'", f"{probs[best_idx]*100:.1f}% confidence")

            st.bar_chart({labels[i]: float(probs[i]) for i in top5})
        else:
            st.info("Start drawing a character on the left canvas.")
    else:
        st.info("Start drawing a character on the left canvas.")

st.divider()
st.markdown(
    "**Roadmap:** this CNN classifies one isolated character at a time. "
    "To extend it to full word/sentence recognition, swap the classifier "
    "head for a CRNN (CNN feature extractor + BiLSTM + CTC loss) so the "
    "model can read a *sequence* of characters from a single image without "
    "needing them pre-segmented — see `crnn_extension.py`."
)
