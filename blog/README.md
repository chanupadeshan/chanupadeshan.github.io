# Blog posts

Each post has its own folder containing its HTML article, Markdown source, and any post-specific assets. Shared styles stay in `article.css`.

```text
blog/
├── article.css
├── README.md
├── logistic-regression/  # English HTML article and Markdown source
├── linear-regression/    # English HTML article and Markdown source
├── knn/                  # English HTML article and Markdown source
├── supervised-vs-unsupervised-learning.html  # redirect for the old URL
├── data-preprocessing/
│   ├── examples/
│   │   ├── image.py
│   │   ├── tabular.py
│   │   ├── text.py
│   │   └── time_series.py
│   ├── data-preprocessing.html
│   ├── data-preprocessing.md
│   ├── data-preprocessing.si.html
│   ├── data-preprocessing.si.md
│   └── requirements.txt
├── nlp-preprocessing/
│   ├── examples/nlp_pipeline.py
│   ├── nlp-preprocessing.html
│   ├── nlp-preprocessing.md
│   ├── nlp-preprocessing.si.html
│   ├── nlp-preprocessing.si.md
│   └── requirements.txt
├── neural-network-architecture/
│   ├── examples/mlp_from_scratch.py
│   ├── neural-network-architecture.html
│   ├── neural-network-architecture.md
│   ├── neural-network-architecture.si.html
│   ├── neural-network-architecture.si.md
│   └── requirements.txt
└── supervisedVSunsupervised/
    ├── supervised-vs-unsupervised-learning.html
    ├── supervised-vs-unsupervised-learning.md
    ├── supervised-vs-unsupervised-learning.si.html
    └── supervised-vs-unsupervised-learning.si.md
```

## Posts

- [Logistic Regression: How It Works for Classification](logistic-regression/logistic-regression.html) · [Markdown source](logistic-regression/logistic-regression.md)
- [Linear Regression: How Machines Predict Continuous Values](linear-regression/linear-regression.html) · [Markdown source](linear-regression/linear-regression.md)
- [K-Nearest Neighbors (KNN) Explained](knn/knn.html) · [Markdown source](knn/knn.md)

- [Neural network architecture: layers, shapes, and learning](neural-network-architecture/neural-network-architecture.html)
- [Neural network architecture Sinhala version](neural-network-architecture/neural-network-architecture.si.html)
- [Neural network architecture Markdown sources](neural-network-architecture/neural-network-architecture.md)
- [Runnable NumPy MLP](neural-network-architecture/examples/mlp_from_scratch.py) and [requirements](neural-network-architecture/requirements.txt)
- [End-to-end NLP preprocessing: from raw text to predictions](nlp-preprocessing/nlp-preprocessing.html)
- [NLP preprocessing Sinhala version](nlp-preprocessing/nlp-preprocessing.si.html)
- [NLP preprocessing Markdown source](nlp-preprocessing/nlp-preprocessing.md)
- [Runnable NLP pipeline](nlp-preprocessing/examples/nlp_pipeline.py) and [requirements](nlp-preprocessing/requirements.txt)
- [Data preprocessing: from raw data to model-ready inputs](data-preprocessing/data-preprocessing.html)
- [Data preprocessing Sinhala version](data-preprocessing/data-preprocessing.si.html)
- [Data preprocessing Markdown sources](data-preprocessing/data-preprocessing.md)
- [Supervised Learning vs. Unsupervised Learning](supervisedVSunsupervised/supervised-vs-unsupervised-learning.html)
- [Markdown source](supervisedVSunsupervised/supervised-vs-unsupervised-learning.md)

The website serves HTML directly. No build step is required. Keep the Markdown source and HTML article text in sync when editing.

For another post, create a folder with its Markdown source and HTML page, then add a post card in the root `index.html`. Lowercase folder names with hyphens are a useful convention for future posts; the current folder name is valid and has been preserved.

From a post folder, link to shared styles with `../article.css` and to the homepage and favicon with `../../index.html` and `../../favicon.png`. Update the article's canonical and Open Graph URLs to match its location. Keep a redirect if an existing public URL changes.

## Languages

Earlier posts are available in English and Sinhala. Each language version has its own Markdown source, language metadata, and canonical URL. Bilingual posts include alternate-language links and a language switch. The logistic regression, linear regression, and KNN posts are English-only. All pages load the same navigation styles from `../../styles/navigation.css`.

Keep both language versions in sync when changing the article. Sinhala uses Noto Sans Sinhala with system font fallbacks. Language links are ordinary links and work without JavaScript.
