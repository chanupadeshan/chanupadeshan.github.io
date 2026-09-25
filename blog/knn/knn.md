---
title: "K-Nearest Neighbors (KNN) Explained"
description: "Understand how KNN uses distance, neighbor votes, feature scaling, and the choice of k for classification and regression."
author: "Chanupa Deshan"
language: en
date: 2026-09-21
---

# K-Nearest Neighbors (KNN) Explained

Imagine identifying a fruit by comparing its weight and size with fruits you have already labeled. If the most similar examples are apples, an apple is a reasonable prediction. K-nearest neighbors turns that intuition into an algorithm: find nearby training examples and combine their answers.

KNN supports both classification and regression. It does not learn a single equation of coefficients like linear or logistic regression. Instead, the stored training examples are central to making each prediction.

## KNN in one minute

For every new example, KNN follows the same simple recipe:

1. Measure the distance from the new example to the training examples.
2. Select the nearest `k` examples.
3. For classification, take their majority vote. For regression, average their values.

The `k` tells us how many neighbors to consult. It is a setting chosen before prediction, not the number of classes in the data.

## How one prediction works

For a new example, KNN measures its distance from training examples, selects the `k` closest, and combines their labels or target values. The letter `k` is a hyperparameter you choose, not the number of classes.

Suppose we classify fruit using two already-standardized features. A new fruit lies at `(0, 0)`. Consider these four training examples:

| Fruit | Feature coordinates | Euclidean distance to the new fruit |
| --- | --- | --- |
| Apple | (0.3, 0.4) | 0.50 |
| Orange | (0.0, 0.6) | 0.60 |
| Apple | (0.6, 0.8) | 1.00 |
| Orange | (1.2, 0.9) | 1.50 |

At `k = 3`, the nearest neighbors are apple, orange, and apple. Uniform voting predicts **apple**, with two of the three votes. The fourth example does not contribute to this prediction.

The vote fraction `2/3` can be reported as an estimated class probability. It is a local vote proportion, not a guarantee that the new fruit has a precisely calibrated 66.7% probability of being an apple.

## What does “near” mean?

The usual starting point is Euclidean distance. For two features:

```text
distance(x, q) = √((x₁ - q₁)² + (x₂ - q₂)²)
```

For the first fruit, that is `√(0.3² + 0.4²) = 0.5`. In more dimensions, add one squared difference per feature before taking the square root.

Other distance measures encode different ideas of similarity. Manhattan distance sums absolute differences. The appropriate representation and metric depend on the task; arbitrary integer codes for categories can create meaningless distances. Encoding apple as `1`, orange as `2`, and banana as `3` does not make banana twice as far from apple as orange is.

Distance also determines the shape of neighborhoods. Choosing a metric is part of choosing the model, rather than a cosmetic implementation detail.

## Why feature scaling matters

Suppose the features are weight in grams and diameter in centimeters. Weight differences may be tens of units while diameter differences are only a few. Raw Euclidean distance can therefore be dominated by weight, even if diameter is equally useful.

Standardization transforms each feature using its training mean and standard deviation:

```text
scaled value = (value - training mean) / training standard deviation
```

Fit the scaler on training data only, and apply that same transformation to new examples. In cross-validation, fit it separately within each training fold. A pipeline makes that sequence easier to get right.

Scaling does not tell us which features matter. Irrelevant features still contribute noise, and outliers can influence the scaling statistics. Feature selection, robust scaling, and an appropriate distance metric may improve the neighborhoods.

## Choosing k: local detail versus smoothing

With `k = 1`, a prediction follows its single closest example. The resulting boundary can be highly detailed and sensitive to noisy labels. Larger values average over wider neighborhoods, making predictions smoother but potentially erasing smaller local patterns.

At the extreme, if `k` equals the number of training examples, uniform classification voting predicts the training set’s majority class everywhere, subject to tie handling. That is usually too much smoothing.

Try several values through validation or cross-validation. There is no universally best `k`. An odd `k` avoids an equal vote count between two classes under uniform voting, but it does not eliminate multiclass ties, equal-distance ties, or all ties under weighted voting.

Distance weighting gives closer neighbors more influence than farther ones. It can help when the nearest examples are especially informative, but it can also amplify a nearby mislabeled example. Scikit-learn offers uniform and distance-weighted voting in its [nearest-neighbors implementation](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification).

## KNN can predict continuous values too

For regression, replace the class vote with an average of neighboring targets. If the three closest deliveries took 20, 24, and 28 minutes, a uniform KNN regressor predicts `(20 + 24 + 28) / 3 = 24` minutes. A distance-weighted regressor gives more influence to closer deliveries.

With nonnegative averaging weights, the prediction stays within the range of the neighbors’ observed targets. KNN therefore does not extrapolate a rising trend the way a fitted straight line can. Beyond the observed region, it continues averaging the closest available examples.

## Try classification in Python

Install with `python -m pip install scikit-learn`. This example uses the bundled Iris dataset, which contains flower measurements and three species. The search chooses `k` and voting weights using only the training partition. The test set remains separate until final evaluation.

```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42,
)
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])
search = GridSearchCV(
    pipeline,
    {"knn__n_neighbors": [3, 5, 7, 11],
     "knn__weights": ["uniform", "distance"]},
    cv=5,
    scoring="accuracy",
)
search.fit(X_train, y_train)
predictions = search.predict(X_test)
print("Selected settings:", search.best_params_)
print("Test accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
```

Because scaling is inside the pipeline, each validation fold uses statistics learned only from its own training fold. After selection, the search refits the chosen pipeline on all training examples. The small test set is useful for a demonstration, but it cannot establish real-world reliability by itself.

## When KNN is useful—and where it struggles

KNN is a useful baseline when similarity is meaningful and the dataset is manageable. Its local predictions can form nonlinear decision boundaries without explicitly building polynomial features.

However, prediction can be expensive: a brute-force search compares a query with all training examples, requiring roughly `O(n × d)` distance work for `n` rows and `d` features. Search trees can help in suitable settings, but their advantage often declines in high dimensions. Storing the training data also consumes memory.

In high-dimensional spaces, examples can become sparse and nearest-versus-farthest distances less informative. This is one part of the curse of dimensionality. Class imbalance and uneven sampling density can also make local votes misleading. Evaluate minority-class precision and recall when those errors matter.

KNN makes a simple idea practical: predict from nearby examples. Its quality depends on what “nearby” means, how many neighbors you use, and whether those neighbors represent the new data well.

For a model with a linear classification boundary, read [logistic regression](../logistic-regression/logistic-regression.html). For continuous predictions from learned coefficients, read [linear regression](../linear-regression/linear-regression.html).
