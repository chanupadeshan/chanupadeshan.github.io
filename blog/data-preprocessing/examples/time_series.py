import numpy as np
import pandas as pd

readings = pd.Series(
    [10, 12, np.nan, 14, 13, 15, 16, 18, 17, 19, 20, 21],
    index=pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
    name="value",
)
# Keep the measured target separate. Do not invent evaluation labels.
features = pd.DataFrame({
    "lag_1": readings.shift(1),
    "past_mean_3": readings.shift(1).rolling(3, min_periods=1).mean(),
    "hour": readings.index.hour,
})
features = features.ffill(limit=1)  # Only past values; fill short gaps.
data = features.join(readings.rename("target")).dropna()
cutoff = pd.Timestamp("2026-01-01 08:00", tz="UTC")
train = data.loc[data.index < cutoff]
test = data.loc[data.index >= cutoff]
print("Training rows:", len(train), "Test rows:", len(test))
print(test.head())
# This is rolling one-step prediction: earlier observations become available.
# For a fixed multi-step forecast, future actual values cannot supply lags.
