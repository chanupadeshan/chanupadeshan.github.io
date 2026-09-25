---
title: "Linear Regression: How Machines Predict Continuous Values"
description: "Learn the line of best fit, coefficients, mean squared error, and R² through a worked example and Python."
author: "Chanupa Deshan"
language: en
date: 2026-09-21
---

# Linear Regression: How Machines Predict Continuous Values

How long will a delivery take? How much electricity will a building use? When the answer is a continuous number, we have a regression problem. Linear regression predicts that number by learning a weighted relationship between inputs and a target.

Its starting point is familiar: a straight line through a scatter plot. The important questions are how we choose that line, what its coefficients mean, and whether it predicts unseen examples well.

## The whole idea in four steps

Picture several dots on a graph. Linear regression searches for a line that follows their overall direction. It then uses that line to predict a value for a new input.

1. Put the input into the line's equation.
2. Multiply each feature by its learned coefficient.
3. Add the intercept to produce a prediction.
4. Measure how far that prediction is from the real value.

Training finds the coefficients that make these errors as small as possible across the training examples. Evaluation checks whether the learned relationship also works on examples the model did not train on.

## The line of best fit

With one input feature, a linear regression model is:

```text
ŷ = b + wx
```

The symbol `ŷ` means the predicted target. The intercept `b` is the prediction when `x = 0`; the slope `w` is the change in prediction for a one-unit increase in `x`.

Imagine a teaching example where delivery time in minutes is predicted from distance in kilometers:

```text
predicted minutes = 12 + 3 × distance
```

For a 6 km trip, the prediction is `12 + 3 × 6 = 30` minutes. The slope says each additional kilometer adds three predicted minutes. The intercept may represent fixed preparation time, but that interpretation needs evidence: a fitted intercept is not automatically a meaningful physical quantity, especially if zero is outside the observed data range.

The **line of best fit** usually means the line minimizing the sum of squared vertical differences between observed and predicted targets. It does not need to pass through every point, and “best” refers to this specific training objective.

## Coefficients with multiple inputs

Real delivery times may depend on distance, package count, and traffic conditions. Multiple linear regression extends the equation:

```text
ŷ = b + w₁x₁ + w₂x₂ + … + wₘxₘ
```

Each coefficient describes a change in the model’s prediction while holding the other input features fixed. A package-count coefficient of `1.5` means one extra package adds 1.5 predicted minutes at the same values of the other inputs.

Coefficient magnitudes depend on units. A coefficient per meter will differ from a coefficient per kilometer, so a larger coefficient does not automatically mean a more important feature. Strongly correlated features can also make individual coefficients unstable. Associations learned from observational data do not establish causation.

With two inputs, the fitted surface is a plane; with more, it is a hyperplane. “Linear” refers to linearity in the coefficients. A model using `x` and `x²` as features can fit a curve while remaining a linear regression model in its parameters.

## MSE: putting a number on the errors

A residual is an observed value minus its prediction: `eᵢ = yᵢ - ŷᵢ`. Mean squared error squares each residual and averages the results:

```text
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

Consider three deliveries and the illustrative line from above:

| Distance (km) | Actual minutes | Predicted minutes | Residual | Squared residual |
| --- | --- | --- | --- | --- |
| 2 | 20 | 18 | 2 | 4 |
| 4 | 22 | 24 | -2 | 4 |
| 6 | 32 | 30 | 2 | 4 |

The MSE is `(4 + 4 + 4) / 3 = 4` minutes squared. Root mean squared error, `RMSE = √MSE`, is two minutes and uses the original target’s units. This illustrative line is not claimed to be the least-squares fit for these three observations.

Squaring prevents positive and negative errors from canceling, but also gives large errors extra influence. One error of ten minutes contributes as much squared error as 25 errors of two minutes each. This is why outliers deserve investigation.

Training ordinary least squares minimizes the sum of squared residuals; minimizing MSE gives the same coefficients because the number of training examples is fixed. Solvers can use numerical linear algebra rather than iterative gradient descent. See the [ordinary least-squares guide](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares).

## R²: comparing with a constant prediction

MSE tells us the error scale, but not how much variation the model captures. For a nonconstant set of observed targets, the coefficient of determination is:

```text
R² = 1 - SS_res / SS_tot
SS_res = Σ(yᵢ - ŷᵢ)²
SS_tot = Σ(yᵢ - ȳ)²
```

Here, `ȳ` is the mean of the observed targets in the set being evaluated. In our three-delivery example, the mean is about `24.67`, `SS_res = 12`, and `SS_tot ≈ 82.67`. Therefore `R² ≈ 0.855`.

- `R² = 1`: predictions match every observed target exactly.
- `R² = 0`: squared error equals that of predicting the evaluation set’s mean for every example.
- `R² < 0`: predictions have more squared error than that constant reference.

An R² of `0.855` does not mean “85.5% accurate.” It describes squared-error reduction relative to the mean reference. On held-out data, R² can be negative. For constant targets, the usual formula has a zero denominator; libraries may apply special handling, as explained in the [R² API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).

The evaluation-set mean in the formula is a scoring reference, not a deployable predictor learned from training. For a practical baseline, also compare against a model that always predicts the **training-set mean**.

## Try it in Python

Install with `python -m pip install scikit-learn`. This synthetic example generates delivery times with a known relationship and random noise, then evaluates on a held-out split.

```python
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = rng.uniform(1, 20, size=(200, 1))  # Distance in kilometers.
y = 12 + 3 * X[:, 0] + rng.normal(0, 4, size=200)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42,
)
model = LinearRegression().fit(X_train, y_train)
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("Test MSE:", mse)
print("Test RMSE:", np.sqrt(mse))
print("Test R²:", r2_score(y_test, predictions))
print("Baseline MSE:", mean_squared_error(
    y_test, baseline.predict(X_test),
))
print("Prediction for 6 km:", model.predict([[6]])[0])
```

The learned intercept and slope should be near 12 and 3, but noise and sampling mean they will not match exactly. This easy synthetic dataset illustrates the mechanics; its score says nothing about performance on real deliveries.

## Check the fit before trusting predictions

Plot residuals against predictions and inputs. Curved patterns suggest missing nonlinear structure; an expanding spread suggests changing error variance. Investigate unusual observations and evaluate the model on data that resembles future use. Chronological data may require a time-based split instead of a random split.

Normal residuals are not required simply to fit a least-squares line. Additional assumptions matter when interpreting conventional confidence intervals and hypothesis tests. For prediction, the central question remains whether errors are acceptable on representative unseen data.

Be especially careful with extrapolation. A relationship learned from 1–20 km deliveries may be unreasonable for a 500 km trip. Linear regression provides a clear numerical baseline, not a promise that the same trend continues forever.

Compare this with [logistic regression](../logistic-regression/logistic-regression.html), which estimates class probabilities, or [KNN](../knn/knn.html), which predicts using nearby examples.
