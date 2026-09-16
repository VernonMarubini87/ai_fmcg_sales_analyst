import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_sales_file
from src.cleaning.validation import run_quality_checks
from src.cleaning.cleaner import clean_dataset

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "fmcg_sales_sample.csv"


def test_quality_checks_detect_injected_duplicate():
    df = load_sales_file(SAMPLE)
    report = run_quality_checks(df)
    dup_issues = [i for i in report["issues"] if i["check"] == "duplicate_rows"]
    assert len(dup_issues) == 1
    assert dup_issues[0]["count"] >= 1


def test_quality_checks_detect_injected_missing_cost():
    df = load_sales_file(SAMPLE)
    report = run_quality_checks(df)
    # cost isn't in the critical-key check list, so missing cost alone
    # shouldn't appear as a "missing_" issue -- confirm no false positive
    missing_issues = [i["check"] for i in report["issues"]]
    assert "missing_cost" not in missing_issues


def test_quality_score_is_high_for_mostly_clean_sample():
    df = load_sales_file(SAMPLE)
    report = run_quality_checks(df)
    # sample has ~16.7k rows, only 1 dup + 50 missing cost (not scored) -> should be near 100
    assert report["quality_score"] > 95


def test_negative_revenue_flagged_as_info_not_dropped():
    df = pd.DataFrame({
        "revenue": [100, -50, 200],
        "quantity": [1, 1, 2],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "product": ["A", "B", "C"],
    })
    report = run_quality_checks(df)
    neg = [i for i in report["issues"] if i["check"] == "negative_revenue"]
    assert len(neg) == 1
    assert neg[0]["severity"] == "info"


def test_future_dates_flagged_critical():
    df = pd.DataFrame({
        "revenue": [100, 200],
        "date": ["2099-01-01", "2026-01-01"],
        "product": ["A", "B"],
    })
    report = run_quality_checks(df)
    future = [i for i in report["issues"] if i["check"] == "future_dates"]
    assert len(future) == 1
    assert future[0]["count"] == 1


def test_cleaner_drops_exact_duplicates():
    df = pd.DataFrame({
        "revenue": [100, 100, 200],
        "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "product": ["A", "A", "B"],
    })
    result = clean_dataset(df)
    assert len(result["cleaned_df"]) == 2
    dedup_actions = [a for a in result["actions"] if a["action"] == "drop_duplicates"]
    assert dedup_actions[0]["count"] == 1


def test_cleaner_removes_unparseable_dates_and_logs_them():
    df = pd.DataFrame({
        "revenue": [100, 200, 300],
        "date": ["2026-01-01", "not_a_date", "2026-01-03"],
        "product": ["A", "B", "C"],
    })
    result = clean_dataset(df)
    assert len(result["cleaned_df"]) == 2
    assert len(result["rejected_rows"]) == 1
    assert result["rejected_rows"].iloc[0]["product"] == "B"


def test_cleaner_coerces_non_numeric_revenue_to_null_not_dropped():
    df = pd.DataFrame({
        "revenue": [100, "bad_value", 300],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "product": ["A", "B", "C"],
    })
    result = clean_dataset(df)
    # row is NOT dropped, just the value becomes null
    assert len(result["cleaned_df"]) == 3
    assert pd.isna(result["cleaned_df"].loc[1, "revenue"])


def test_cleaner_on_real_sample_preserves_row_count_minus_duplicate():
    df = load_sales_file(SAMPLE)
    before = len(df)
    result = clean_dataset(df)
    # only the 1 injected duplicate should be removed; dates/revenue are all valid
    assert len(result["cleaned_df"]) == before - 1
    assert result["rejected_rows"].empty
