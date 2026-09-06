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
