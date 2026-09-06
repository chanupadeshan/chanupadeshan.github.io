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
