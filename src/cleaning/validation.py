"""
Data quality validation: run a battery of FMCG-specific checks against
a dataset and return a structured report. This module only DETECTS
issues — it never modifies data. Fixing happens in cleaner.py, and only
for issues explicitly marked as safe to auto-fix.
"""
from __future__ import annotations
import pandas as pd

from src.ingestion.schema import detect_schema


def run_quality_checks(df: pd.DataFrame) -> dict:
    """
    Run all quality checks and return a report:
    {
        "issues": [ {check, severity, count, description}, ... ],
        "quality_score": 0-100,
        "row_count": int,
    }
    Severity is "critical" (breaks analysis if unaddressed),
    "warning" (should be reviewed), or "info" (cosmetic).
    """
    schema = detect_schema(df)
    issues = []

    issues += _check_duplicates(df)
    issues += _check_missing_keys(df, schema)
    issues += _check_negative_revenue(df, schema)
    issues += _check_negative_quantity(df, schema)
    issues += _check_zero_revenue_with_quantity(df, schema)
    issues += _check_future_dates(df, schema)
    issues += _check_unparseable_dates(df, schema)
    issues += _check_revenue_cost_mismatch(df, schema)

    quality_score = _compute_quality_score(issues, len(df))

    return {
        "issues": issues,
        "quality_score": quality_score,
        "row_count": len(df),
    }


def _check_duplicates(df: pd.DataFrame) -> list[dict]:
    n = int(df.duplicated().sum())
    if n == 0:
        return []
    return [{
        "check": "duplicate_rows",
        "severity": "warning",
        "count": n,
        "description": f"{n} fully duplicated row(s) found.",
    }]


def _check_missing_keys(df: pd.DataFrame, schema: dict) -> list[dict]:
    issues = []
    for role in ("revenue", "date", "product"):
        col = schema.get(role)
        if not col:
            continue
        n = int(df[col].isna().sum())
        if n > 0:
            issues.append({
                "check": f"missing_{role}",
                "severity": "critical",
                "count": n,
                "description": f"{n} row(s) missing '{col}' ({role}).",
            })
    return issues


def _check_negative_revenue(df: pd.DataFrame, schema: dict) -> list[dict]:
    col = schema.get("revenue")
    if not col:
        return []
    negative = pd.to_numeric(df[col], errors="coerce") < 0
    n = int(negative.sum())
    if n == 0:
        return []
    return [{
        "check": "negative_revenue",
        "severity": "info",
        "count": n,
        "description": (
            f"{n} row(s) have negative revenue in '{col}'. "
            "This is often legitimate (credit notes/returns) in FMCG data "
            "rather than an error — verify against a doc_type/credit-note field if present."
        ),
    }]


def _check_negative_quantity(df: pd.DataFrame, schema: dict) -> list[dict]:
    col = schema.get("quantity")
    if not col:
        return []
    negative = pd.to_numeric(df[col], errors="coerce") < 0
    n = int(negative.sum())
    if n == 0:
        return []
    return [{
        "check": "negative_quantity",
        "severity": "warning",
        "count": n,
        "description": f"{n} row(s) have negative quantity in '{col}'.",
    }]


def _check_zero_revenue_with_quantity(df: pd.DataFrame, schema: dict) -> list[dict]:
    rev_col, qty_col = schema.get("revenue"), schema.get("quantity")
    if not rev_col or not qty_col:
        return []
    revenue = pd.to_numeric(df[rev_col], errors="coerce")
    quantity = pd.to_numeric(df[qty_col], errors="coerce")
    mask = (revenue == 0) & (quantity > 0)
    n = int(mask.sum())
    if n == 0:
        return []
    return [{
        "check": "zero_revenue_with_quantity",
        "severity": "warning",
        "count": n,
        "description": (
            f"{n} row(s) have zero revenue but positive quantity — "
            "possibly free samples, write-offs, or a pricing data error."
        ),
    }]


def _check_future_dates(df: pd.DataFrame, schema: dict) -> list[dict]:
    col = schema.get("date")
    if not col:
        return []
    parsed = pd.to_datetime(df[col], errors="coerce")
    future = parsed > pd.Timestamp.now()
    n = int(future.sum())
    if n == 0:
        return []
    return [{
        "check": "future_dates",
        "severity": "critical",
        "count": n,
        "description": f"{n} row(s) have a transaction date in the future.",
    }]


def _check_unparseable_dates(df: pd.DataFrame, schema: dict) -> list[dict]:
    col = schema.get("date")
    if not col:
        return []
    parsed = pd.to_datetime(df[col], errors="coerce")
    n = int(parsed.isna().sum() - df[col].isna().sum())
    if n <= 0:
        return []
    return [{
        "check": "unparseable_dates",
        "severity": "critical",
        "count": n,
        "description": f"{n} row(s) have a date value that could not be parsed.",
    }]


def _check_revenue_cost_mismatch(df: pd.DataFrame, schema: dict) -> list[dict]:
    rev_col, cost_col = schema.get("revenue"), schema.get("cost")
    if not rev_col or not cost_col:
        return []
    revenue = pd.to_numeric(df[rev_col], errors="coerce")
    cost = pd.to_numeric(df[cost_col], errors="coerce")
    mask = (cost > revenue) & revenue.notna() & cost.notna()
    n = int(mask.sum())
    if n == 0:
        return []
    return [{
        "check": "cost_exceeds_revenue",
        "severity": "warning",
        "count": n,
        "description": f"{n} row(s) have cost greater than revenue (loss-making lines) — may be legitimate but worth flagging.",
    }]


def _compute_quality_score(issues: list[dict], row_count: int) -> float:
    """
    Simple, explainable scoring: start at 100, deduct per affected row
    weighted by severity, floor at 0. This is intentionally transparent
    rather than a black-box score, since it feeds into evidence shown to Claude.
    """
    if row_count == 0:
        return 0.0

    weights = {"critical": 3.0, "warning": 1.0, "info": 0.2}
    penalty = 0.0
    for issue in issues:
        weight = weights.get(issue["severity"], 1.0)
        penalty += weight * (issue["count"] / row_count) * 100

    score = max(0.0, 100.0 - penalty)
    return round(score, 1)
