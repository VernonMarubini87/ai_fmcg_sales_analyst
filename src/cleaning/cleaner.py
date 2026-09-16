"""
Data cleaning: applies only the fixes that are safe to automate.
Everything this module does is logged so nothing is silently discarded.

Safe-to-automate actions:
- drop exact duplicate rows
- coerce the date column to datetime, dropping rows that fail to parse
  (they're logged, and also returned separately so they can be inspected)
- coerce revenue/quantity/cost columns to numeric

NOT automated (flagged only, left for human/analyst review):
- negative revenue (often legitimate credit notes)
- cost > revenue
- zero revenue with positive quantity
"""
from __future__ import annotations
import pandas as pd

from src.ingestion.schema import detect_schema


def clean_dataset(df: pd.DataFrame) -> dict:
    """
    Returns:
        {
            "cleaned_df": pd.DataFrame,
            "rejected_rows": pd.DataFrame,   # rows removed, for inspection
            "actions": [ {action, count, description}, ... ],
        }
    """
    schema = detect_schema(df)
    working = df.copy()
    actions = []
    rejected_frames = []

    working, action = _drop_duplicates(working)
    if action:
        actions.append(action)

    working, rejected, action = _coerce_dates(working, schema)
    if action:
        actions.append(action)
    if not rejected.empty:
        rejected_frames.append(rejected)

    working, action = _coerce_numeric(working, schema, "revenue")
    if action:
        actions.append(action)

    working, action = _coerce_numeric(working, schema, "quantity")
    if action:
        actions.append(action)

    working, action = _coerce_numeric(working, schema, "cost")
    if action:
        actions.append(action)

    rejected_rows = (
        pd.concat(rejected_frames).drop_duplicates()
        if rejected_frames else working.iloc[0:0]
    )

    return {
        "cleaned_df": working.reset_index(drop=True),
        "rejected_rows": rejected_rows.reset_index(drop=True),
        "actions": actions,
    }


def _drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    before = len(df)
    deduped = df.drop_duplicates()
    n_removed = before - len(deduped)
    if n_removed == 0:
        return df, None
    return deduped, {
        "action": "drop_duplicates",
        "count": n_removed,
        "description": f"Removed {n_removed} exact duplicate row(s).",
    }


def _coerce_dates(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    date_col = schema.get("date")
    if not date_col:
        return df, df.iloc[0:0], None

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    bad_mask = parsed.isna() & df[date_col].notna()
    n_bad = int(bad_mask.sum())

    rejected = df[bad_mask].copy()
    kept = df[~bad_mask].copy()
    kept[date_col] = parsed[~bad_mask]

    if n_bad == 0:
        return kept, rejected, None

    return kept, rejected, {
        "action": "drop_unparseable_dates",
        "count": n_bad,
        "description": f"Removed {n_bad} row(s) with a date that could not be parsed in '{date_col}'.",
    }


def _coerce_numeric(df: pd.DataFrame, schema: dict, role: str) -> tuple[pd.DataFrame, dict | None]:
    col = schema.get(role)
    if not col:
        return df, None

    original = df[col]
    coerced = pd.to_numeric(original, errors="coerce")
    newly_null = int((coerced.isna() & original.notna()).sum())

    df = df.copy()
    df[col] = coerced

    if newly_null == 0:
        return df, None

    return df, {
        "action": f"coerce_{role}_numeric",
        "count": newly_null,
        "description": (
            f"{newly_null} non-numeric value(s) in '{col}' ({role}) were converted to null "
            "rather than dropped — they still appear as missing values, not deleted rows."
        ),
    }
