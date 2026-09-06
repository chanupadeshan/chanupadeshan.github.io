---
title: "Data preprocessing: from raw data to model-ready inputs"
description: "What data preprocessing is, why it matters, and practical Python examples for tables, text, images, and time series."
author: "Chanupa Deshan"
language: en
date: 2026-09-05
---

# Data preprocessing: from raw data to model-ready inputs

A model does not see a customer, a sentence, or a photograph the way we do. It receives a representation of that information. Data preprocessing is how we turn raw observations into consistent inputs that a model can use.

In the [previous post](../supervisedVSunsupervised/supervised-vs-unsupervised-learning.html), we compared supervised and unsupervised learning. Both depend on the quality of those inputs. Here, we will prepare several kinds of data and write the Python code behind each step.

## What is preprocessing, and why does it matter?

Preprocessing includes checking data types, correcting known errors, handling missing values, and converting inputs into a useful representation. Feature engineering is closely related: it creates useful signals, such as the hour of a transaction or yesterday’s temperature.

Imagine a price column containing `1200`, `"1,200"`, an empty cell, and an amount recorded in a different currency. Converting everything to a number is not enough. You also need to understand units, what a blank means, and which values are valid.

Good preprocessing makes inputs consistent, helps suitable models learn efficiently, and makes training and real-world prediction follow the same rules. It can also expose collection errors before they become model errors. It does not fix incorrect labels, missing populations, or a poorly defined problem on its own.

| Data type | Typical preparation | A mistake to avoid |
| --- | --- | --- |
| Tables | Imputation, encoding, scaling | Fitting statistics on the test set |
| Text | Unicode handling, tokenization, vectorization | Removing meaning with aggressive cleaning |
| Images | Orientation, color mode, resizing, pixel conversion | Applying the wrong pretrained-model transform |
| Time series | Timestamp alignment, lags, temporal splits | Using future observations in current features |
| Audio and video | Sample/frame rates, alignment, feature extraction | Splitting clips from one recording across train and test |

## A workflow that keeps evaluation honest

Inspect the schema and define the prediction task first. Decide which information will actually exist when the prediction is made. Resolve known duplicate records and group related samples so the same person, document, or recording does not leak across splits.

Then create training, validation, and test sets suited to the task. A random split can work for independent examples; grouped or chronological splits are needed when records are related or ordered in time.

**Fit learned preprocessing on training data only.** Medians, means, category vocabularies, and feature-selection rules all learn from data. Reuse those fitted transformations for validation, testing, and production. In cross-validation, fit them again inside each training fold. A pipeline helps keep preprocessing and the model together. See scikit-learn’s [guide to data leakage](https://scikit-learn.org/stable/common_pitfalls.html).

The examples use Python and small teaching datasets. Download [requirements.txt](requirements.txt), then install the packages in a virtual environment:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download each linked Python file and run it, for example `python tabular.py`. These examples demonstrate transformations, not production model quality.

## Tabular data: missing values, categories, and scale

Start by validating types, units, ranges, and duplicate identifiers. Investigate outliers before deleting them: a large transaction may be the very event you want to detect. Missing values may require a median, a category such as “unknown,” a missingness indicator, or a model that handles them directly. Choose based on what absence means.

For many linear and distance-based models, numeric scaling matters. Standardization subtracts the training mean and divides by the training standard deviation; min–max scaling maps the training range to a chosen interval. Robust scaling uses statistics such as the median and interquartile range. Tree-based models usually do not need the same scaling. These are alternatives, not a checklist to apply together. See [scikit-learn’s preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html).

One-hot encoding represents categories as indicator columns without inventing a numeric order. Use ordinal encoding only when the order is meaningful. High-cardinality categories may need a different representation; target encoding needs particular care to avoid leakage.

This example imputes and scales numeric columns, encodes cities, and trains a classifier. The pipeline accepts an unseen city without crashing, but ignoring an unknown category does not guarantee a good prediction. The pattern uses [ColumnTransformer and Pipeline](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html).

[Download tabular.py](examples/tabular.py)

```python
"""Small teaching example; predictions are not a quality benchmark."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = pd.DataFrame({
    "age": [22, 45, np.nan, 31, 52, 27, 39, 24, 48, 35, 29, 56],
    "income": [30, 80, 45, np.nan, 95, 35, 65, 28, 88, 55, 40, 100],
    "city": ["Kandy", "Colombo", "Galle"] * 4,
})
y = [0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

numeric = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])
prepare = ColumnTransformer([
    ("numeric", numeric, ["age", "income"]),
    ("category", categorical, ["city"]),
])
model = Pipeline([
    ("prepare", prepare),
    ("predict", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)  # Learn every statistic on training data only.
print("Test predictions:", model.predict(X_test))
new_customer = pd.DataFrame({
    "age": [33], "income": [50], "city": ["Jaffna"]
})
print("New category prediction:", model.predict(new_customer))
```

## Text: preserve meaning before creating numbers

Text preprocessing is task-dependent. Common steps include decoding correctly, removing unwanted markup, normalizing Unicode, and tokenization. Lowercasing, stop-word removal, stemming, and lemmatization are optional. Removing “not” can reverse sentiment; deleting punctuation can lose useful signals. Stripping every non-English character would destroy Sinhala text.

Traditional models can use counts or TF-IDF features. Here, character n-grams provide a simple baseline for English and Sinhala without assuming that an English word tokenizer works for both. TF-IDF learns a vocabulary and inverse-document-frequency weights from the training corpus. A sparse output matrix stores mostly nonzero entries efficiently. The [TfidfVectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) explains its options.

[Download text.py](examples/text.py)

```python
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    # Keep Sinhala characters, punctuation, and words such as "not".
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


train_texts = [
    "This service is good!", "This service is not good.",
    "මෙම සේවාව හොඳයි", "මෙම සේවාව හොඳ නැහැ",
]
test_texts = ["  The service\n is good. ", "සේවාව හොඳයි"]
vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), lowercase=True
)
X_train = vectorizer.fit_transform([clean_text(t) for t in train_texts])
X_test = vectorizer.transform([clean_text(t) for t in test_texts])
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
# Both matrices use the vocabulary and IDF learned from train_texts.
```

The test matrix has the same number of feature columns as the training matrix. It does not grow a new vocabulary from the test messages. Character n-grams are not a complete linguistic tokenizer. For transformer models, use the tokenizer associated with the checkpoint, including its padding, truncation, and attention-mask conventions. Avoid cleaning away information that tokenizer expects.

## Images: consistent shape, color, and pixel values

First check that images decode and that labels are correct. Respect camera orientation, choose a consistent color mode, and decide whether to resize, crop, or pad. Stretching changes geometry; cropping can remove the subject; padding adds borders. The best choice depends on the task.

The example preserves aspect ratio, pads to a square, converts to RGB, and maps an 8-bit image into a float array in `[0, 1]`. Its output layout is height × width × channels. Some models expect channels first or additional normalization. Follow the exact preprocessing of a pretrained checkpoint. See [Pillow’s image operations](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html) and [Torchvision’s transform guide](https://docs.pytorch.org/vision/stable/transforms.html).

[Download image.py](examples/image.py)

```python
import numpy as np
from PIL import Image, ImageOps


def prepare_image(image):
    image = ImageOps.exif_transpose(image)  # Respect camera orientation.
    image = image.convert("RGB")
    image = ImageOps.pad(
        image, (224, 224), method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
    )  # Preserve aspect ratio and add padding.
    return np.asarray(image, dtype=np.float32) / 255.0


# A synthetic input keeps the example runnable without downloading a photo.
example = Image.new("RGB", (320, 180), color=(80, 160, 240))
pixels = prepare_image(example)
print("Shape:", pixels.shape)  # (224, 224, 3): height, width, channels
print("Type:", pixels.dtype)  # float32
print("Range:", pixels.min(), pixels.max())  # Within [0, 1]
# For your photo:
# with Image.open("photo.jpg") as photo:
#     pixels = prepare_image(photo)
```

This conversion is for ordinary 8-bit RGB images, not a universal recipe for medical scans or high-bit-depth sensor data. Dividing by 255 is pixel scaling; per-channel mean/std normalization is a separate operation.

Training-time augmentation adds plausible variations such as crops, brightness changes, or flips. Keep validation/test preprocessing deterministic unless you deliberately evaluate a defined test-time augmentation scheme. A horizontal flip may preserve a cat label but corrupt written text. For segmentation and detection, transform masks or boxes consistently with the image.

## Time series: prepare the past, not the future

Parse timestamps, standardize time zones, sort observations, and check gaps or irregular sampling. Resampling can place readings on a shared interval; use an aggregation suited to the measurement. Missing observations are not automatically zeros.

For forecasting, create lagged values and rolling statistics that use only information available at the prediction time. Backward filling or interpolation using later observations can leak the future. The example below fills only short feature gaps from the past, leaves missing targets unfilled, and separates training from testing by time. Relevant APIs are covered in the [pandas time-series guide](https://pandas.pydata.org/docs/user_guide/timeseries.html).

[Download time_series.py](examples/time_series.py)

```python
import numpy as np
import pandas as pd

readings = pd.Series(
    [10, 12, np.nan, 14, 13, 15, 16, 18, 17, 19, 20, 21],
    index=pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
    name="value",
)
# Keep the measured target separate. Do not invent evaluation labels.
features = pd.DataFrame({
    "lag_1": readings.shift(1),
    "past_mean_3": readings.shift(1).rolling(3, min_periods=1).mean(),
    "hour": readings.index.hour,
})
features = features.ffill(limit=1)  # Only past values; fill short gaps.
data = features.join(readings.rename("target")).dropna()
cutoff = pd.Timestamp("2026-01-01 08:00", tz="UTC")
train = data.loc[data.index < cutoff]
test = data.loc[data.index >= cutoff]
print("Training rows:", len(train), "Test rows:", len(test))
print(test.head())
# This is rolling one-step prediction: earlier observations become available.
# For a fixed multi-step forecast, future actual values cannot supply lags.
```

The output has six training rows and four test rows. This is a **rolling one-step** setup: by the time you predict the next reading, earlier readings are observed. For a forecast of the entire next day made at midnight, actual values from later that day cannot supply lag features. Match the code to your forecasting horizon.

## What about audio, video, and imbalanced data?

For audio, check the sample rate, channel count, clipping, and recording quality. You may resample, select or combine channels, trim irrelevant silence, divide recordings into windows, or compute spectrograms and MFCCs. A speech model may already include some of these steps. Do not discard channel information if location or direction matters, and do not normalize away loudness when it carries the target signal.

For video, choose a frame-sampling policy, apply compatible image transforms, and preserve temporal order. Keep synchronized audio, frames, and labels aligned. Split by source recording or subject when nearby clips would otherwise be nearly identical across train and test.

Class imbalance is another concern across data types. Consider class weights, suitable metrics, and possibly training-set resampling. If you oversample or create synthetic examples, do it only within the training partition of each fold. Do not rebalance the held-out test set just to make scores look better.

## A practical checklist

- Check types, units, missingness, duplicates, label quality, and representation of relevant groups.
- Choose a split that matches the way the model will encounter new data.
- Fit learned transformations on training data only; preserve the fitted pipeline.
- Validate shapes, ranges, unknown categories, empty text, broken files, and missing timestamps.
- Apply the same inference preprocessing during testing and deployment.
- Compare against a simple baseline and check whether each transformation helps the actual task.
- Record package versions and preprocessing choices alongside the model.

Preprocessing is part of the model’s behavior. Treat it as code you can inspect, test, and reuse—not a collection of cleaning steps you have to repeat manually.
