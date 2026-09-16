from __future__ import annotations
import pandas as pd
import numpy as np

from src.ingestion.schema import detect_schema
from src.analytics.sales import daily_sales


def rolling_sales_anomalies(df: pd.DataFrame, window: int = 7, threshold: float = 2.0) -> pd.DataFrame | None:
    """
    Flags days where revenue deviates from its own rolling mean by more
    than `threshold` standard deviations. This is deliberately relative
    to recent history rather than the whole-period mean, since FMCG sales
    have strong weekly/seasonal patterns that a flat mean would misread.
    """
    daily = daily_sales(df)
    if daily is None or len(daily) < window + 1:
        return None

    result = daily.copy()
    result["rolling_mean"] = result["revenue"].rolling(window).mean()
    result["rolling_std"] = result["revenue"].rolling(window).std()
    result["upper_threshold"] = result["rolling_mean"] + threshold * result["rolling_std"]
    result["lower_threshold"] = result["rolling_mean"] - threshold * result["rolling_std"]
    result["is_anomaly"] = (
        (result["revenue"] > result["upper_threshold"])
        | (result["revenue"] < result["lower_threshold"])
    )
    return result


def zscore_anomalies(series: pd.Series, threshold: float = 3.0) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce")
    mean, std = values.mean(), values.std()

    result = pd.DataFrame({"value": values})
    if std == 0 or pd.isna(std):
        result["z_score"] = 0.0
        result["is_z_anomaly"] = False
        return result

    result["z_score"] = (values - mean) / std
    result["is_z_anomaly"] = result["z_score"].abs() >= threshold
    return result


def detect_anomalies(df: pd.DataFrame) -> dict:
    """Top-level anomaly summary combining daily-revenue rolling detection."""
    rolling = rolling_sales_anomalies(df)
    if rolling is None:
        return {"available": False, "reason": "Insufficient daily revenue history for anomaly detection."}

    flagged = rolling[rolling["is_anomaly"]]
    return {
        "available": True,
        "anomaly_count": int(len(flagged)),
        "anomalies": flagged[["period", "revenue", "rolling_mean"]].to_dict("records"),
    }
