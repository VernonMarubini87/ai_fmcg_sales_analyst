from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.analytics.sales import daily_sales
from src.forecasting.baseline import seasonal_naive_forecast, mae, rmse, mape


def _build_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy().sort_values("period").reset_index(drop=True)
    df["dayofweek"] = df["period"].dt.dayofweek
    df["day"] = df["period"].dt.day
    df["month"] = df["period"].dt.month
    for lag in (1, 7, 14):
        df[f"lag_{lag}"] = df["revenue"].shift(lag)
    df["rolling_mean_7"] = df["revenue"].shift(1).rolling(7).mean()
    return df


def train_and_forecast(df: pd.DataFrame, horizon: int = 30, test_size: int = 30) -> dict:
    """
    Trains a Random Forest on lag/calendar features to forecast daily
    revenue, validates against a held-out tail of history, and compares
    against a seasonal-naive baseline. Returns available=False if there
    isn't enough history to do this responsibly.
    """
    daily = daily_sales(df)
    if daily is None or len(daily) < (test_size + 30):
        return {"available": False, "reason": "Not enough daily revenue history to forecast reliably (need 60+ days)."}

    features_df = _build_features(daily).dropna().reset_index(drop=True)
    feature_cols = ["dayofweek", "day", "month", "lag_1", "lag_7", "lag_14", "rolling_mean_7"]

    train = features_df.iloc[:-test_size]
    test = features_df.iloc[-test_size:]

    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)
    model.fit(train[feature_cols], train["revenue"])
    predictions = model.predict(test[feature_cols])

    baseline_forecast = seasonal_naive_forecast(train["revenue"], periods=len(test), season_length=7)

    evaluation = {
        "model": {
            "mae": mae(test["revenue"], predictions),
            "rmse": rmse(test["revenue"], predictions),
            "mape": mape(test["revenue"], predictions),
        },
        "seasonal_naive_baseline": {
            "mae": mae(test["revenue"].values, baseline_forecast.values),
            "rmse": rmse(test["revenue"].values, baseline_forecast.values),
            "mape": mape(test["revenue"].values, baseline_forecast.values),
        },
    }

    # Recursive forecast forward from the end of full history
    history = features_df.copy()
    future_rows = []
    last_date = history["period"].max()

    working = daily.copy().sort_values("period").reset_index(drop=True)
    for step in range(horizon):
        next_date = last_date + pd.Timedelta(days=step + 1)
        recent = working["revenue"]
        row = {
            "period": next_date,
            "dayofweek": next_date.dayofweek,
            "day": next_date.day,
            "month": next_date.month,
            "lag_1": recent.iloc[-1],
            "lag_7": recent.iloc[-7] if len(recent) >= 7 else recent.iloc[-1],
            "lag_14": recent.iloc[-14] if len(recent) >= 14 else recent.iloc[-1],
            "rolling_mean_7": recent.iloc[-7:].mean(),
        }
        pred = float(model.predict(pd.DataFrame([row])[feature_cols])[0])
        future_rows.append({"period": next_date, "forecast_revenue": pred})
        working = pd.concat([working, pd.DataFrame([{"period": next_date, "revenue": pred}])], ignore_index=True)

    forecast_df = pd.DataFrame(future_rows)

    return {
        "available": True,
        "evaluation": evaluation,
        "forecast": forecast_df,
        "forecast_total": float(forecast_df["forecast_revenue"].sum()),
        "horizon_days": horizon,
    }
