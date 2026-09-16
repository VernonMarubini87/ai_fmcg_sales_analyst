from __future__ import annotations
import pandas as pd
import numpy as np


def naive_forecast(series: pd.Series, periods: int) -> pd.Series:
    last_value = series.iloc[-1]
    return pd.Series([last_value] * periods)


def seasonal_naive_forecast(series: pd.Series, periods: int, season_length: int = 7) -> pd.Series:
    """Default season_length=7 for daily FMCG data (weekly pattern)."""
    if len(series) < season_length:
        return naive_forecast(series, periods)
    seasonal_values = series.iloc[-season_length:].values
    forecast = [seasonal_values[i % season_length] for i in range(periods)]
    return pd.Series(forecast)


def mae(actual, predicted) -> float:
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted) -> float:
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual, predicted) -> float:
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
