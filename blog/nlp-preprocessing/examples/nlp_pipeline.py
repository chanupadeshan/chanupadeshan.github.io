"""Run with Python 3.10+; synthetic data demonstrates mechanics, not quality."""
import argparse
import json
import platform
import unicodedata
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def normalize_text(text):
    if not isinstance(text, str):
        raise TypeError("Expected a string")
    text = " ".join(unicodedata.normalize("NFC", text).split())
    if not text:
        raise ValueError("Empty text requires review")
    return text


def prepare_records(records):
    """Validate labels and remove normalized exact duplicates before splitting."""
    unique = {}
    for text, label in records:
        if label not in {"positive", "negative"}:
            raise ValueError(f"Unsupported label: {label}")
        text = normalize_text(text)
        if text in unique and unique[text] != label:
            raise ValueError("Conflicting labels for duplicate text")
        unique[text] = label
    return list(unique), list(unique.values())


def main(output_dir):
    # Replace these independent examples with audited (text, label) records.
    positive = [
        "Excellent service!", "I love this product.", "Delivery was fast.",
        "The staff were helpful.", "A wonderful experience.", "Very happy with it.",
        "Everything works perfectly.", "Great quality for the price.",
        "I would buy this again.", "Support solved my problem.",
        "The order arrived safely.", "The app is easy to use.",
        "This is really good.", "I recommend this shop.",
        "The result exceeded expectations.", "A reliable service.",
        "Setup was quick and simple.", "I enjoyed using it.",
        "The team did a fantastic job.", "Thank you for the excellent help.",
    ]
    negative = [
        "Terrible service!", "I hate this product.", "Delivery was late.",
        "The staff were unhelpful.", "An awful experience.", "Very disappointed with it.",
        "Nothing works correctly.", "Poor quality for the price.",
        "I would never buy this again.", "Support ignored my problem.",
        "The order arrived broken.", "The app is difficult to use.",
        "This is not good.", "I cannot recommend this shop.",
        "The result failed expectations.", "An unreliable service.",
        "Setup was slow and confusing.", "I regret using it.",
        "The team did a dreadful job.", "Still waiting for help.",
    ]
    records = [(t, "positive") for t in positive]
    records += [(t, "negative") for t in negative]
    texts, labels = prepare_records(records)
    # 60% train, 20% validation, 20% test; independent examples assumed.
    train_x, hold_x, train_y, hold_y = train_test_split(
        texts, labels, test_size=0.4, stratify=labels, random_state=42
    )
    val_x, test_x, val_y, test_y = train_test_split(
        hold_x, hold_y, test_size=0.5, stratify=hold_y, random_state=42
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize_text, lowercase=False,
            analyzer="char_wb", ngram_range=(3, 5), min_df=1,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(train_x, train_y)
    print("Split sizes:", len(train_x), len(val_x), len(test_x))
    print("Validation (use for development):")
    print(classification_report(val_y, model.predict(val_x), zero_division=0))
    # In a real project, freeze choices before this one-time final evaluation.
    print("Test (tiny synthetic sample, not a benchmark):")
    print(classification_report(test_y, model.predict(test_x), zero_division=0))
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "sentiment.joblib")
    metadata = {
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "joblib": joblib.__version__, "normalization": "NFC + whitespace v1",
        "seed": 42, "data": "synthetic teaching corpus v1",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # Only load trusted artifacts. Keep this script and matching dependencies.
    restored = joblib.load(output_dir / "sentiment.joblib")
    incoming = ["  This is not good!\n", "Excellent help, thank you!"]
    assert list(restored.predict(incoming)) == list(model.predict(incoming))
    print("New predictions:", restored.predict(incoming))
    print("Artifacts:", output_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    main(parser.parse_args().output_dir)
