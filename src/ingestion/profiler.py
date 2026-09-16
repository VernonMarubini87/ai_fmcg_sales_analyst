"""
Data profiling: produce an evidence-grade summary of the uploaded dataset
before anything gets analysed or shown to Claude. This is the foundation
the data-quality (Phase 2) and QA (Phase 13) layers build on.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from .schema import detect_schema, schema_summary


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Build a full profile of the dataset: shape, schema detection,
    per-column stats, missing values, duplicates, and date coverage.
    """
    schema = detect_schema(df)
    schema_info = schema_summary(schema)

    profile = {
        "row_count": len(df),
        "column_count": df.shape[1],
        "columns": list(df.columns),
        "schema": schema_info,
        "missing_values": _missing_values(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_stats": _column_stats(df),
    }

    date_col = schema.get("date")
    if date_col:
        profile["date_coverage"] = _date_coverage(df, date_col)

    return profile


def _missing_values(df: pd.DataFrame) -> dict[str, dict]:
    result = {}
    total = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            result[col] = {
                "missing_count": n_missing,
                "missing_pct": round(n_missing / total * 100, 2) if total else 0.0,
            }
    return result


def _column_stats(df: pd.DataFrame) -> dict[str, dict]:
    stats = {}
    for col in df.columns:
        series = df[col]
        entry = {"dtype": str(series.dtype)}

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            entry.update({
                "min": float(clean.min()) if len(clean) else None,
                "max": float(clean.max()) if len(clean) else None,
                "mean": float(clean.mean()) if len(clean) else None,
                "negative_count": int((clean < 0).sum()) if len(clean) else 0,
                "zero_count": int((clean == 0).sum()) if len(clean) else 0,
            })
        else:
            entry["unique_values"] = int(series.nunique(dropna=True))

        stats[col] = entry
    return stats


def _date_coverage(df: pd.DataFrame, date_col: str) -> dict:
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    valid = parsed.dropna()
    unparseable = int(parsed.isna().sum() - df[date_col].isna().sum())

    if valid.empty:
        return {"min_date": None, "max_date": None, "unparseable_dates": unparseable}

    return {
        "min_date": valid.min().strftime("%Y-%m-%d"),
        "max_date": valid.max().strftime("%Y-%m-%d"),
        "distinct_dates": int(valid.dt.date.nunique()),
        "unparseable_dates": unparseable,
    }
