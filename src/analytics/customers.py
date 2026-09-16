from __future__ import annotations
import pandas as pd
from src.ingestion.schema import detect_schema
from src.analytics.products import _agg_by


def customer_performance(df: pd.DataFrame) -> pd.DataFrame | None:
    schema = detect_schema(df)
    return _agg_by(df, schema.get("customer"), schema.get("revenue"), schema.get("quantity"))


def rfm_analysis(df: pd.DataFrame) -> pd.DataFrame | None:
    schema = detect_schema(df)
    customer_col, date_col, rev_col = schema.get("customer"), schema.get("date"), schema.get("revenue")
    tx_col = schema.get("transaction_id")

    if not all([customer_col, date_col, rev_col]):
        return None

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    if temp.empty:
        return None

    analysis_date = temp[date_col].max() + pd.Timedelta(days=1)

    if tx_col:
        frequency = temp.groupby(customer_col)[tx_col].nunique()
    else:
        frequency = temp.groupby(customer_col).size()

    rfm = temp.groupby(customer_col).agg(
        recency=(date_col, lambda x: (analysis_date - x.max()).days),
        monetary=(rev_col, "sum"),
    )
    rfm["frequency"] = frequency
    return rfm.reset_index()
