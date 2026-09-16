from __future__ import annotations
import pandas as pd
from src.ingestion.schema import detect_schema
from src.analytics.products import _agg_by


def store_performance(df: pd.DataFrame) -> pd.DataFrame | None:
    schema = detect_schema(df)
    return _agg_by(df, schema.get("store"), schema.get("revenue"), schema.get("quantity"))


def regional_performance(df: pd.DataFrame) -> pd.DataFrame | None:
    schema = detect_schema(df)
    return _agg_by(df, schema.get("region"), schema.get("revenue"), schema.get("quantity"))
