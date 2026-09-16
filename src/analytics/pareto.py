from __future__ import annotations
import pandas as pd
from src.ingestion.schema import detect_schema


def pareto_analysis(df: pd.DataFrame, dimension_role: str) -> pd.DataFrame | None:
    """
    dimension_role: one of the canonical schema roles, e.g. "product", "customer", "store".
    """
    schema = detect_schema(df)
    dim_col = schema.get(dimension_role)
    rev_col = schema.get("revenue")

    if not dim_col or not rev_col:
        return None

    result = (
        df.groupby(dim_col, dropna=False)[rev_col]
        .sum()
        .reset_index()
        .rename(columns={rev_col: "revenue"})
        .sort_values("revenue", ascending=False)
    )

    total = result["revenue"].sum()
    if total == 0:
        return result

    result["contribution_pct"] = result["revenue"] / total * 100
    result["cumulative_contribution_pct"] = result["contribution_pct"].cumsum()
    result["pareto_80"] = result["cumulative_contribution_pct"] <= 80
    return result.reset_index(drop=True)
