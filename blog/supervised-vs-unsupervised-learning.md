---
title: "Supervised Learning vs. Unsupervised Learning"
description: "How labels change what a machine learning model can learn, with practical examples and guidance on choosing an approach."
author: "Chanupa Deshan"
status: published
date: 2026-09-05
tags:
  - Machine Learning
  - Supervised Learning
  - Unsupervised Learning
---

# Supervised Learning vs. Unsupervised Learning

Imagine you have a spreadsheet containing thousands of bank transactions. You want to use machine learning to make sense of them.

If past transactions have been marked as fraudulent or legitimate, you can train a model to predict that label for new transactions. If you only have transaction details, you might instead look for groups of similar transactions or unusual patterns.

These are examples of supervised and unsupervised learning. The main difference is whether training data includes a target answer for the task.

## Supervised learning: learning from labeled examples

In supervised learning, each training example includes input features and a target. The model learns a relationship between them so it can predict targets for unseen inputs.

For a house-price model, the features might be floor area, location, and number of bedrooms. The target is the recorded sale price.

| Input features | Target |
| --- | --- |
| Floor area, location, bedrooms | House sale price |
| Email text and sender information | Spam or legitimate |
| Transaction amount, time, and merchant | Fraudulent or legitimate |

The two common task types are:

- **Classification:** predicting a category, such as whether an email is spam.
- **Regression:** predicting a numeric value, such as a house price.

Common algorithms include linear regression, logistic regression, decision trees, random forests, and support vector machines. Neural networks can also be trained for supervised tasks.

During training, an algorithm adjusts the model to reduce a loss: a measure of how far its predictions are from the known targets. But performing well on training examples is only part of the job. We evaluate on held-out data to check whether the model generalizes.

The evaluation metric should match the task. For house prices, mean absolute error expresses the average error in the same currency as the target. For rare fraud cases, accuracy alone can be misleading: predicting every transaction as legitimate may appear accurate while missing every fraudulent transaction. Precision and recall help reveal that failure.

## Unsupervised learning: finding structure without target labels

In unsupervised learning, the training data does not provide target labels for the task. The algorithm looks for structure according to its objective and assumptions.

Consider a bank that wants to explore customer behavior. It has features such as transaction frequency, average spending, and account balance, but no predefined customer-segment labels.

A clustering algorithm can group customers with similar feature values. An analyst then examines those groups and decides whether they are useful. Names such as “frequent small purchases” are human interpretations of the results.

Two common task types are:

- **Clustering:** grouping similar examples. K-means and DBSCAN are common algorithms.
- **Dimensionality reduction:** representing data with fewer dimensions. Principal component analysis (PCA) is one approach, which preserves as much variance as possible for a chosen number of components.

Unsupervised methods can also help flag unusual observations. An unusual transaction deserves investigation, but unusual behavior alone does not prove fraud.

Without target labels, evaluation takes more judgment. You can examine cluster separation, stability, and usefulness for the intended task. A good numerical clustering score does not automatically mean you have found meaningful customer segments.

## The key differences

| Question | Supervised learning | Unsupervised learning |
| --- | --- | --- |
| What training data is available? | Features with target labels | Features without target labels for the task |
| What is the goal? | Predict a known target | Discover or represent structure |
| Example | Predict whether a customer will leave | Group customers by behavior |
| Common tasks | Classification and regression | Clustering and dimensionality reduction |
| How is it evaluated? | Predictions against held-out targets | Structure, stability, and practical usefulness |
| Main challenge | Obtaining reliable, representative labels | Deciding whether discovered patterns are meaningful |

## Which should you use?

Start with the question you want to answer.

If you want to predict a specific outcome and have reliable historical labels, supervised learning is a natural starting point. If you want to explore patterns without a predefined target, unsupervised learning may help.

Both depend on how you prepare the data. For example, k-means uses distances, so a feature measured in thousands can dominate another measured between zero and one unless you handle scaling appropriately. Supervised models can also produce misleading results if their inputs contain information that would only become available after the prediction is needed. That is data leakage.

The approaches can work together. You might first explore customer groups, then train a supervised model to predict customer churn. Any preprocessing used in that predictive pipeline should be fitted on the training data to keep evaluation honest.

Before choosing an algorithm, write down your target, the data available at prediction time, and what a useful result would look like. Those decisions will guide the approach more reliably than starting with a favorite model.
