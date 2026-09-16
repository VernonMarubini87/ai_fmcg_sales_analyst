from __future__ import annotations
import pandas as pd
from src.ingestion.schema import detect_schema


def _prepare_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out.dropna(subset=[date_col])


def _period_sales(df: pd.DataFrame, freq_label: str) -> pd.DataFrame | None:
    schema = detect_schema(df)
    date_col, rev_col, qty_col = schema.get("date"), schema.get("revenue"), schema.get("quantity")

    if not date_col or not rev_col:
        return None

    temp = _prepare_dates(df, date_col)
    if temp.empty:
        return None

    if freq_label == "day":
        temp["period"] = temp[date_col].dt.date
        temp["period"] = pd.to_datetime(temp["period"])
    elif freq_label == "week":
        temp["period"] = temp[date_col].dt.to_period("W").apply(lambda x: x.start_time)
    elif freq_label == "month":
        temp["period"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
    else:
        raise ValueError(f"Unknown freq_label {freq_label}")

    aggregation = {rev_col: "sum"}
    if qty_col:
        aggregation[qty_col] = "sum"

    result = temp.groupby("period").agg(aggregation).reset_index()
    rename_map = {rev_col: "revenue"}
    if qty_col:
        rename_map[qty_col] = "units"
    result = result.rename(columns=rename_map)
    return result.sort_values("period").reset_index(drop=True)


def daily_sales(df: pd.DataFrame) -> pd.DataFrame | None:
    return _period_sales(df, "day")


def weekly_sales(df: pd.DataFrame) -> pd.DataFrame | None:
    return _period_sales(df, "week")


def monthly_sales(df: pd.DataFrame) -> pd.DataFrame | None:
    return _period_sales(df, "month")


def calculate_monthly_growth(df: pd.DataFrame) -> pd.DataFrame | None:
    result = monthly_sales(df)
    if result is None:
        return None
    result = result.copy()
    result["revenue_mom_growth_pct"] = result["revenue"].pct_change() * 100
    if "units" in result.columns:
        result["volume_mom_growth_pct"] = result["units"].pct_change() * 100
    return result


def calculate_yoy_growth(df: pd.DataFrame) -> pd.DataFrame | None:
    result = monthly_sales(df)
    if result is None:
        return None
    result = result.copy()
    result["revenue_yoy_growth_pct"] = result["revenue"].pct_change(periods=12) * 100
    if "units" in result.columns:
        result["volume_yoy_growth_pct"] = result["units"].pct_change(periods=12) * 100
    return result


def volume_price_analysis(df: pd.DataFrame) -> dict | None:
    """Decompose revenue into volume vs average-price components (period totals)."""
    schema = detect_schema(df)
    rev_col, qty_col = schema.get("revenue"), schema.get("quantity")
    if not rev_col or not qty_col:
        return None

    total_revenue = float(pd.to_numeric(df[rev_col], errors="coerce").sum())
    total_units = float(pd.to_numeric(df[qty_col], errors="coerce").sum())
    if total_units == 0:
        return None

    return {
        "total_revenue": total_revenue,
        "total_units": total_units,
        "average_price": total_revenue / total_units,
    }
