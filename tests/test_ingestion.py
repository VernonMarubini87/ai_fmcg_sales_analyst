import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_sales_file
from src.ingestion.schema import detect_schema, find_column
from src.ingestion.profiler import profile_dataset

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "fmcg_sales_sample.csv"


def test_load_normalizes_column_names():
    df = load_sales_file(SAMPLE)
    assert "sales_value" in df.columns
    assert "product_name" in df.columns
    assert not any(" " in c for c in df.columns)


def test_schema_detection_finds_core_roles():
    df = load_sales_file(SAMPLE)
    schema = detect_schema(df)
    assert schema["revenue"] == "sales_value"
    assert schema["quantity"] == "quantity"
    assert schema["date"] == "sales_date"
    assert schema["product"] == "product_name"
    assert schema["store"] == "store"
    assert schema["region"] == "region"
    assert schema["target"] == "sales_target"
    assert schema["cost"] == "cost"


def test_schema_detection_missing_role_is_none():
    df = pd.DataFrame({"revenue": [1, 2, 3]})
    schema = detect_schema(df)
    assert schema["date"] is None
    assert schema["customer"] is None


def test_find_column_case_insensitive():
    df = pd.DataFrame({"Net_Sales": [1, 2]})
    df.columns = df.columns.str.lower()
    assert find_column(df, ["net_sales", "revenue"]) == "net_sales"


def test_profile_dataset_shape_and_duplicates():
    df = load_sales_file(SAMPLE)
    profile = profile_dataset(df)
    assert profile["row_count"] == len(df)
    assert profile["duplicate_rows"] >= 1  # we injected one


def test_profile_flags_missing_cost():
    df = load_sales_file(SAMPLE)
    profile = profile_dataset(df)
    assert "cost" in profile["missing_values"]
    assert profile["missing_values"]["cost"]["missing_count"] == 50


def test_profile_date_coverage():
    df = load_sales_file(SAMPLE)
    profile = profile_dataset(df)
    coverage = profile["date_coverage"]
    assert coverage["min_date"] == "2025-01-01"
    assert coverage["max_date"] == "2026-08-31"


def test_profile_negative_and_zero_counts_present_for_numeric():
    df = load_sales_file(SAMPLE)
    profile = profile_dataset(df)
    assert "negative_count" in profile["column_stats"]["quantity"]
