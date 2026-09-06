# Blog posts

Each post has its own folder containing its HTML article, Markdown source, and any post-specific assets. Shared styles stay in `article.css`.

```text
blog/
├── article.css
├── README.md
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
└── supervisedVSunsupervised/
    ├── supervised-vs-unsupervised-learning.html
    ├── supervised-vs-unsupervised-learning.md
    ├── supervised-vs-unsupervised-learning.si.html
    └── supervised-vs-unsupervised-learning.si.md
```

## Posts

- [Data preprocessing: from raw data to model-ready inputs](data-preprocessing/data-preprocessing.html)
- [Data preprocessing Sinhala version](data-preprocessing/data-preprocessing.si.html)
- [Data preprocessing Markdown sources](data-preprocessing/data-preprocessing.md)
- [Supervised Learning vs. Unsupervised Learning](supervisedVSunsupervised/supervised-vs-unsupervised-learning.html)
- [Markdown source](supervisedVSunsupervised/supervised-vs-unsupervised-learning.md)

The website serves HTML directly. No build step is required. Keep the Markdown source and HTML article text in sync when editing.

For another post, create a folder with its Markdown source and HTML page, then add a post card in the root `index.html`. Lowercase folder names with hyphens are a useful convention for future posts; the current folder name is valid and has been preserved.

From a post folder, link to shared styles with `../article.css` and to the homepage and favicon with `../../index.html` and `../../favicon.png`. Update the article's canonical and Open Graph URLs to match its location. Keep a redirect if an existing public URL changes.

## Languages

Both posts are available in English and Sinhala. Each has its own Markdown source, language metadata, canonical URL, alternate-language links, and a language switch. All pages load the same navigation styles from `../../styles/navigation.css`.

Keep both language versions in sync when changing the article. Sinhala uses Noto Sans Sinhala with system font fallbacks. Language links are ordinary links and work without JavaScript.
