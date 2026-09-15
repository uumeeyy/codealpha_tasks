# Handwritten Character Recognition

CNN-based recognizer for handwritten digits and characters, built on
MNIST and EMNIST-balanced, with a live drawing-canvas demo and a
skeleton for extending to full word/sentence recognition (CRNN).

## Project structure

```
handwritten_char_recognition/
├── model.py              # CNN architecture (shared by digits & chars)
├── train.py               # Trains on MNIST or EMNIST-balanced
├── predict.py             # CLI inference on a single image file
├── app.py                  # Streamlit demo: draw a character, see live prediction
├── crnn_extension.py       # CNN+BiLSTM+CTC skeleton for word/sentence recognition
├── requirements.txt
└── models/                 # Populated after training (.keras files, labels, plots)
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train

Digits only (MNIST, 10 classes, fast — a couple minutes/epoch on CPU):

```bash
python train.py --dataset mnist --epochs 15
```

Digits + letters (EMNIST-balanced, 47 classes — first run downloads
~536MB via `tensorflow-datasets` and caches it):

```bash
python train.py --dataset emnist --epochs 15
```

This saves `models/<dataset>_cnn.keras`, `models/<dataset>_labels.txt`,
and an accuracy/loss curve PNG. Expect roughly **99%+ test accuracy on
MNIST** and **~90%+ on EMNIST-balanced** (EMNIST is harder — it's not
purely a digit classification benchmark and includes visually similar
upper/lowercase letters merged into shared classes).

> **Note on this sandbox:** dataset downloads in the sandbox I built
> this in are blocked by network policy (Keras/TFDS fetch from
> `storage.googleapis.com`, which isn't on the allowed-domains list
> here), so I verified the model architecture and the full training
> loop (augmentation → `tf.data` pipeline → `fit`) with synthetic data
> instead of running a real epoch. On your machine or in Colab, where
> that URL is reachable, `train.py` will download the real dataset and
> train normally — no code changes needed.

## 3. Predict on a single image

```bash
python predict.py --image my_letter.png --dataset emnist --topk 3
```

`predict.py` handles arbitrary input photos/scans: it auto-inverts if
needed, crops to the character's bounding box, and resizes to 28×28 to
match the training distribution — this preprocessing matters a lot for
real-world accuracy, since MNIST/EMNIST characters are already tightly
cropped and centered.

## 4. Interactive demo

```bash
streamlit run app.py
```

Draw a digit or letter on the canvas and see the live prediction plus
a bar chart of the top-5 candidate classes. Switch between "Digits
only" and "Digits + Letters" modes with the radio button (each needs
its corresponding model trained first).

## 5. Extending to words/sentences (CRNN)

The CNN here classifies **one pre-segmented character** per image.
Real handwritten words don't segment cleanly (uneven spacing, joined
strokes), so `crnn_extension.py` sketches the standard fix:

```
image → CNN feature extractor → reshape into a left-to-right sequence
      → BiLSTM (bidirectional context) → per-timestep softmax → CTC loss
```

CTC (Connectionist Temporal Classification) loss lets the network
learn its own alignment between image columns and output characters,
so no manual per-character segmentation is required. This file defines
the architecture (`build_crnn`) and greedy CTC decoding
(`ctc_greedy_decode`); wiring it to a dataset is left open since that
depends on which corpus you use — the standard choice is the **IAM
Handwriting Database** for real handwritten words/lines, optionally
pre-training on synthetic words assembled by concatenating random
EMNIST characters.

## Design notes

- **Augmentation** (small rotation/translation/zoom) during training
  meaningfully improves robustness to messy real-world handwriting
  versus training on raw MNIST/EMNIST alone.
- **BatchNorm + Dropout** throughout the CNN head keeps the ~440K-parameter
  model from overfitting despite being deep enough for high accuracy.
- **EarlyStopping + ReduceLROnPlateau** avoid needing to hand-tune the
  epoch count or learning rate schedule.
