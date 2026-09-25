# End-to-end NLP preprocessing: from raw text to predictions

Chanupa Deshan · September 6, 2026 · Practical guide + Python pipeline

Preprocessing decides what an NLP model gets to see. A cleaner sentence is not always a better input: remove “not” from “not good” and you change the label the sentence should receive.

In the [data preprocessing guide](../data-preprocessing/data-preprocessing.html), we introduced text features. Here we build the complete path for document classification: inspect records, preserve meaning, prevent leakage, create features, evaluate, and reuse the fitted pipeline. We also explain where transformer preprocessing takes a different route.

## 01 · Define the task and input contract

Start with the prediction: for example, classify a customer message as positive or negative. Decide whether the model receives a whole document, a sentence, or a conversation. A sentiment classifier, search engine, and named-entity recognizer need different preprocessing choices.

Keep the raw text and a stable record ID. Useful fields include `text`, `label`, `source_id`, `language`, and `created_at`. Retain only metadata you are authorized to use, and separate audit metadata from model features. A review rating used to create a sentiment label must not also become an input feature.

Specify the contract before cleaning: text must be a decoded string, labels must belong to the supported set, and missing or blank inputs need an explicit policy. Do not convert `None` into the literal word “None.” For the example below, invalid or empty text raises an error so it can be reviewed.

## 02 · Inspect, deduplicate, and split

Inspect missing values, label counts, text lengths, languages, repeated records, and annotation disagreements. Check representative examples from every source. Broken decoding, copied templates, and automatically generated signatures can dominate a corpus without being useful signals.

Decode files using their documented encoding, commonly UTF-8. Investigate decoding errors instead of silently discarding bytes. Extract fields from JSON or CSV with their proper parsers. For HTML documents, use an HTML parser to extract the intended visible content; regular expressions are not a general HTML parser. Preserve paragraph boundaries when they matter.

Resolve exact duplicates and conflicting labels using a documented policy. Group near-duplicates, messages from one conversation, or documents from the same source so related content cannot cross partitions. Deterministic normalization can help identify duplicates before splitting; it must not learn vocabulary or statistics from the held-out text.

Use training data to fit the system, validation data to choose settings, and a test set for the final evaluation. Split by customer or conversation when samples are related; split by time when deployment predicts future messages. Stratified random splitting is appropriate only for sufficiently independent examples with enough records per class.

Keep learned preprocessing inside the model pipeline, including during cross-validation. This prevents validation text from influencing feature learning. See the [scikit-learn data leakage guide](https://scikit-learn.org/stable/common_pitfalls.html).

## 03 · Normalize without erasing meaning

A conservative starting point is Unicode NFC normalization plus whitespace cleanup. NFC reconciles canonically equivalent character sequences; it does not translate text or repair incorrect decoding. Compatibility normalization, NFKC, can change distinctions such as full-width characters, so choose it deliberately. See [Python’s Unicode normalization documentation](https://docs.python.org/3/library/unicodedata.html).

```python
import unicodedata


def normalize_text(text):
    if not isinstance(text, str):
        raise TypeError("Expected a string")
    text = " ".join(unicodedata.normalize("NFC", text).split())
    if not text:
        raise ValueError("Empty text requires review")
    return text


print(normalize_text("  This is not good!\n"))
# This is not good!
print(normalize_text("  මෙම සේවාව හොඳ නැහැ  "))
# මෙම සේවාව හොඳ නැහැ
```

Whitespace collapse suits this document-classification example. It would destroy layout information in poetry, source code, or a document extraction task. Likewise, do not strip non-ASCII characters: that removes Sinhala and many other writing systems. Avoid deleting combining marks or zero-width characters indiscriminately.

Treat every additional transform as a hypothesis to evaluate on validation data:

| Choice | When it may help | What it can lose |
| --- | --- | --- |
| Lowercasing | Reducing vocabulary variation in a classical model | Names, acronyms, emphasis, case distinctions |
| URL or username placeholders | Reducing incidental identifiers | Domains or mentions useful for the task |
| Number normalization | Grouping variable numeric values | Prices, dates, quantities, version numbers |
| Punctuation or emoji removal | Removing known extraction noise | Sentiment, sentence boundaries, expressive cues |
| Spelling correction | A domain with verified correction rules | Names, dialects, transliteration, technical terms |
| Stop-word removal | Some retrieval or topic baselines | Negation and useful phrase structure |
| Stemming or lemmatization | Reducing word-form variation | Meaning distinctions and language-specific detail |

Stemming uses rules to shorten word forms and can produce nonwords. Lemmatization aims for a dictionary form and may require language and part-of-speech information. Neither is mandatory. Do not apply an English stemmer to Sinhala. Mixed-language text needs suitable multilingual tools or a tested language-routing policy.

## 04 · Tokenize and choose a representation

Tokenization divides text into units. Word tokens are easy to inspect, character n-grams capture short character sequences, and subword tokenizers split text using a model vocabulary. Splitting on spaces alone is not a universal linguistic tokenizer.

For a classical baseline, word unigrams and bigrams can represent terms and phrases such as “not good.” Character n-grams offer another starting point for spelling variation and multilingual text without an English word segmenter. They still require evaluation for each language; preserving Sinhala characters does not make a model trained only on English reliable in Sinhala.

TF-IDF combines term frequency with inverse document frequency, reducing the relative weight of terms appearing across many training documents. The vectorizer learns its vocabulary and IDF on training text; `transform` reuses those columns for validation and inference. Keep the resulting sparse matrix sparse instead of converting a large corpus to a dense array. The [TfidfVectorizer reference](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) documents these options.

Our baseline uses `char_wb` n-grams of length 3–5, which work within whitespace-delimited words with boundary padding. It preserves case and does not remove stop words. `min_df=1` accommodates the tiny example; larger corpora may benefit from vocabulary limits selected on validation data. Unknown character sequences add no new feature columns; an input with no known features produces an all-zero vector, whose prediction is driven by the classifier’s intercept.

## 05 · Run the complete classical pipeline

Download [nlp_pipeline.py](examples/nlp_pipeline.py) and [requirements.txt](requirements.txt) into the same folder. The script includes 40 synthetic English messages, validates records, removes normalized exact duplicates, creates a 24/8/8 split, fits TF-IDF and logistic regression, reports metrics, and saves and reloads the model. These small, related teaching phrases demonstrate execution, not a credible estimate of real-world accuracy.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python nlp_pipeline.py --output-dir artifacts
```

The central training code is below. `normalize_text` is the function from the previous section; the downloaded script supplies the records and split variables.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=normalize_text, lowercase=False,
        analyzer="char_wb", ngram_range=(3, 5), min_df=1,
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(train_x, train_y)
validation_predictions = model.predict(val_x)
new_predictions = model.predict(["  This is not good!\n"])
```

Supplying a custom vectorizer preprocessor replaces its built-in preprocessing, so casing and normalization belong in that function. The same function runs during training and prediction. Dataset validation and duplicate resolution happen separately before the split; they are not learned feature transformations.

The script reports per-class precision, recall, F1, and support. Compare against a majority-class baseline on real data, and inspect the confusion matrix and errors. Use validation results to compare character versus word features or optional cleaning. Freeze choices before inspecting test results; repeatedly changing rules to improve test scores turns the test set into development data.

For a real dataset, replace the synthetic records with parsed `(text, label)` pairs and implement the group or time split appropriate to the source. Review near-duplicates, multilingual coverage, and label quality before treating a score as useful evidence.

## 06 · The transformer branch

A pretrained transformer needs the tokenizer associated with its checkpoint. Feed minimally processed text into that tokenizer instead of applying TF-IDF or an unrelated stemmer. Its normalization, vocabulary, special tokens, and token IDs are part of the model contract.

This separate illustration requires `transformers` and internet access for the first tokenizer download. It prepares inputs only; it does not train a sentiment classifier. The BERT checkpoint shown is an English uncased example, not a Sinhala model.

```python
from transformers import AutoTokenizer

checkpoint = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
encoded = tokenizer(
    ["This is not good!", "Excellent service."],
    padding=True,
    truncation=True,
    max_length=128,
    return_attention_mask=True,
)
print(encoded["input_ids"])
print(encoded["attention_mask"])
```

Padding makes sequences within this batch equal in length; truncation caps long inputs at the chosen token limit. Here, 128 is a demonstration budget, not a universal optimum. Attention masks identify padding positions. Follow the model’s supported context length and the [Hugging Face padding and truncation guide](https://huggingface.co/docs/transformers/main/pad_truncation).

Measure truncation rates and inspect whether important content falls off the end. For long documents, consider overlapping chunks or a suitable longer-context model, then define how chunk predictions combine. Split source documents before chunking so overlapping passages cannot leak across sets. For token classification, preserve character offsets and align labels with subwords; document labels cannot simply be reused as token labels.

Save the tokenizer alongside the trained model, record the checkpoint revision, and reuse the same special-token and length settings at inference. A pretrained tokenizer normally uses its existing vocabulary; training a new tokenizer on your corpus should use only the training partition.

## 07 · Package the pipeline for inference

The runnable example saves `sentiment.joblib` and `metadata.json` in the output directory. It reloads the artifact and checks that predictions match. Keep the exact source and dependency versions with the artifact; the provided requirements specify compatible ranges, not a reproducible lockfile. Capture the tested environment with `python -m pip freeze > environment.txt`.

Python serialization references the custom normalization function. The example reloads in the same script, where that function is available. For a separate serving application, place it in an importable, versioned module shared by training and serving, then regenerate the artifact. Load only artifacts from a trusted source because pickle-based loading can execute code.

A service should validate request types and sizes, route empty inputs to the documented fallback, and pass accepted raw strings to the fitted pipeline. Do not fit a new vectorizer per request. Test unseen words, negation, punctuation-only input, Unicode variants, and unusually long messages. Treat an all-zero feature vector as an observable condition rather than evidence that a prediction is meaningful.

## 08 · Monitor and improve deliberately

Track empty-input and rejection rates, input length distributions, language mix, zero-feature rates for TF-IDF, and truncation rates for transformers. Monitor these by source where possible. Avoid logging raw private messages when aggregate measurements are sufficient.

Prediction drift alone does not tell you whether accuracy changed. Collect reviewed labels and evaluate per-class and per-language performance over time. Version changes to the corpus, split policy, normalization, vocabulary or tokenizer, and model together so results can be reproduced and rolled back.

Before using the pipeline, check:

- The input contract and empty-text policy are explicit.
- Related records remain in the same split, and future data does not leak backward.
- Unicode, negation, and task-relevant information survive cleaning.
- Learned transformations fit only on training data.
- The representation matches the model and the languages being served.
- Validation guides changes, and the final test set remains held out.
- Reloaded artifacts reproduce predictions with the shared preprocessing code.
- Monitoring can reveal new input patterns and preprocessing failures.

An end-to-end pipeline is a repeatable path from an accepted raw input to a prediction. The useful measure of a cleaning step is whether it improves that system on representative held-out data while preserving the information the task needs.
