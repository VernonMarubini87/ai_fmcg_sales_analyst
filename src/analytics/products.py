from __future__ import annotations
import pandas as pd
from src.ingestion.schema import detect_schema


def _agg_by(df: pd.DataFrame, dim_col: str, rev_col: str | None, qty_col: str | None) -> pd.DataFrame | None:
    if not dim_col:
        return None
    aggregations = {}
    if rev_col:
        aggregations[rev_col] = "sum"
    if qty_col:
        aggregations[qty_col] = "sum"
    if not aggregations:
        return None

    result = df.groupby(dim_col, dropna=False).agg(aggregations).reset_index()
    rename_map = {}
    if rev_col:
        rename_map[rev_col] = "revenue"
    if qty_col:
        rename_map[qty_col] = "units"
    result = result.rename(columns=rename_map)

    if "revenue" in result.columns:
        total = result["revenue"].sum()
        if total != 0:
            result["revenue_contribution_pct"] = result["revenue"] / total * 100
        result = result.sort_values("revenue", ascending=False)

    return result.reset_index(drop=True)


def product_performance(df: pd.DataFrame) -> pd.DataFrame | None:
    schema = detect_schema(df)
    return _agg_by(df, schema.get("product"), schema.get("revenue"), schema.get("quantity"))


def category_performance(df: pd.DataFrame) -> pd.DataFrame | None:
    schema = detect_schema(df)
    return _agg_by(df, schema.get("category"), schema.get("revenue"), schema.get("quantity"))


def product_growth(df: pd.DataFrame) -> pd.DataFrame | None:
    """Month-over-month revenue growth per product."""
    schema = detect_schema(df)
    product_col, date_col, rev_col = schema.get("product"), schema.get("date"), schema.get("revenue")

    if not all([product_col, date_col, rev_col]):
        return None

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    temp["month"] = temp[date_col].dt.to_period("M").dt.to_timestamp()

    result = (
        temp.groupby([product_col, "month"])[rev_col]
        .sum()
        .reset_index()
        .rename(columns={rev_col: "revenue"})
    )
    result["growth_pct"] = result.groupby(product_col)["revenue"].pct_change() * 100
    return result
