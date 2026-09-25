---
title: "Logistic Regression: How It Works for Classification"
description: "Understand the sigmoid function, predicted probabilities, classification thresholds, and decision boundaries with worked examples."
author: "Chanupa Deshan"
language: en
date: 2026-09-21
---

# Logistic Regression: How It Works for Classification

Will a customer cancel a subscription? Is a message spam? These questions ask for a category rather than a continuous number. Logistic regression approaches binary classification by estimating the probability of one of two outcomes.

Despite its name, logistic regression is a classification model. Its useful trick is to turn a weighted sum of input features into a number between zero and one. A separate decision rule turns that probability into a class label.

## The whole idea in three steps

You can understand the model as a short pipeline:

1. Combine the input features into a **score**.
2. Pass the score through the sigmoid to get a **probability**.
3. Compare the probability with a threshold to choose a **class**.

For example, the model might estimate an `0.82` probability that a student will pass. With a threshold of `0.5`, the final prediction is “pass.” The probability is the model's confidence-like estimate; the class is the decision made from it.

## Start with a linear score

Suppose we want to predict whether a student passes an exam, using study hours as the only feature. Encode a pass as `1` and a fail as `0`. Our illustrative model begins with:

```text
z = b + w × hours
z = -4 + 0.8 × hours
```

Here, `b` is the intercept and `w` is the coefficient. Each additional study hour increases the score by `0.8`. These values are invented for the explanation; in a real model, training learns them from labeled examples.

With several features, the score becomes `z = b + w₁x₁ + w₂x₂ + … + wₘxₘ`. The score itself is not a probability: it can be any real number.

## The sigmoid turns scores into probabilities

The sigmoid function maps the score to the model’s estimated probability of class `1`:

```text
p = P(y = 1 | x) = 1 / (1 + exp(-z))
```

The function has an S-shaped curve. A large negative score produces a probability near zero, a score of zero produces exactly `0.5`, and a large positive score produces a probability near one.

| Study hours | Score z | Estimated probability of passing |
| --- | --- | --- |
| 2 | -2.4 | 0.083 |
| 5 | 0.0 | 0.500 |
| 8 | 2.4 | 0.917 |

For eight hours, the model estimates roughly a 92% chance of passing. That is an estimate conditional on the features and learned model, not a guarantee. Whether predictions near 0.9 actually succeed about 90% of the time is a question of probability calibration that should be checked on representative held-out data.

There is also a useful interpretation behind the formula: `log(p / (1 - p)) = z`. The model is linear in **log-odds**, not in probability. Adding an hour changes log-odds by `0.8`; it does not add 80 percentage points to the probability.

## A threshold turns probability into a decision

Choose a threshold `t`, then use the explicit rule:

```text
predict class 1 if p >= t
predict class 0 otherwise
```

At `t = 0.5`, the two-hour student is classified as a fail and the eight-hour student as a pass. A prediction of `0.62` is class `1` at a threshold of `0.5`, but class `0` at a threshold of `0.7`.

Lowering the threshold flags more examples as positive. On a fixed dataset, this cannot reduce recall, but it can create more false positives. Raising it flags fewer positives and can miss more actual positives. Precision does not necessarily change monotonically.

Choose the threshold using validation data and the costs of mistakes. For spam filtering, an incorrectly blocked important message may be more costly than a spam message reaching the inbox. Leave the final test set untouched while making that choice. Scikit-learn’s [threshold guide](https://scikit-learn.org/stable/modules/classification_threshold.html) explains this separation between probability estimation and decision-making.

## Where is the decision boundary?

A decision boundary is the set of inputs where the class decision changes. At a probability threshold of `0.5`, sigmoid reaches the threshold when `z = 0`.

For our example, `-4 + 0.8 × hours = 0`, so the boundary is five hours. With two features and a score `z = -4 + 0.8x₁ + 0.5x₂`, the boundary is the straight line:

```text
0.8x₁ + 0.5x₂ = 4
```

With more features, it is a hyperplane. The S-shaped sigmoid does **not** make this boundary curved in the original features. Adding nonlinear features such as `x₁²` can make the boundary nonlinear in the original input space.

For any threshold strictly between zero and one, the boundary satisfies `z = log(t / (1 - t))`. Changing the threshold moves the boundary without retraining the coefficients.

## How does the model learn?

Training chooses coefficients that make observed labels likely. For binary targets, the usual objective is average binary cross-entropy, also called log loss:

```text
loss = -(1/n) × Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

If the true label is `1`, predicting `0.9` gives a loss of about `0.105`, while predicting `0.1` gives about `2.303`. Confident wrong predictions receive a large penalty. An optimizer adjusts the coefficients to reduce this loss.

Regularization penalizes overly large coefficients and can help generalization. Scikit-learn’s [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) applies regularization by default; its `C` parameter controls inverse regularization strength, so smaller values mean stronger regularization.

## Try it in Python

This example creates a synthetic binary dataset, fits scaling only on the training data, and applies a fixed threshold. Install the dependency with `python -m pip install scikit-learn`. The dataset is for teaching, not a performance benchmark.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=600, n_features=4, n_informative=3,
    n_redundant=0, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42,
)
model = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000),
)
model.fit(X_train, y_train)
# Classes are [0, 1], so column 1 is the probability of class 1.
probabilities = model.predict_proba(X_test)[:, 1]
threshold = 0.5  # Fixed in advance; tune only on validation data.
predictions = (probabilities >= threshold).astype(int)
print(classification_report(y_test, predictions, zero_division=0))
print("Log loss:", log_loss(y_test, probabilities))
print("First five probabilities:", probabilities[:5].round(3))
```

The classification report evaluates decisions at the chosen threshold; log loss evaluates the probabilities. Neither should be replaced with training accuracy. With imbalanced classes, inspect precision and recall for the class you care about instead of relying only on overall accuracy.

## What to remember

Logistic regression combines three distinct steps: a linear score, a sigmoid probability, and a threshold-based decision. Its speed and interpretable coefficients make it a useful baseline, but feature quality, a suitable boundary, and honest evaluation still matter.

For continuous targets, read [Linear Regression: How Machines Predict Continuous Values](../linear-regression/linear-regression.html). For a classifier built around similarity, continue with [K-Nearest Neighbors (KNN) Explained](../knn/knn.html).
