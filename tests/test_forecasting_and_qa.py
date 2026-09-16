import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_sales_file
from src.forecasting.baseline import naive_forecast, seasonal_naive_forecast, mae, rmse, mape
from src.forecasting.models import train_and_forecast
from src.qa.analytical_qa import validate_numeric, validate_percentage, validate_kpis
from src.qa.evidence import build_evidence, validate_evidence

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "fmcg_sales_sample.csv"


# ---------- Baseline forecasting ----------

def test_naive_forecast_repeats_last_value():
    series = pd.Series([10, 20, 30])
    result = naive_forecast(series, periods=3)
    assert list(result) == [30, 30, 30]


def test_seasonal_naive_forecast_repeats_pattern():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7])
    result = seasonal_naive_forecast(series, periods=7, season_length=7)
    assert list(result) == [1, 2, 3, 4, 5, 6, 7]


def test_evaluation_metrics():
    actual = [100, 200, 300]
    predicted = [110, 190, 300]
    assert mae(actual, predicted) == pytest_approx(20 / 3)  # (|100-110|+|200-190|+|300-300|)/3
    assert rmse(actual, predicted) > 0
    assert mape(actual, predicted) > 0


def pytest_approx(x, tol=1e-6):
    class _Approx(float):
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx(x)


# ---------- ML forecasting on real sample ----------

def test_train_and_forecast_on_real_sample():
    df = load_sales_file(SAMPLE)
    result = train_and_forecast(df, horizon=14, test_size=30)
    assert result["available"] is True
    assert len(result["forecast"]) == 14
    assert result["forecast_total"] > 0
    assert "mae" in result["evaluation"]["model"]
    assert "mae" in result["evaluation"]["seasonal_naive_baseline"]


def test_train_and_forecast_unavailable_with_short_history():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10),
        "revenue": range(10),
    })
    result = train_and_forecast(df)
    assert result["available"] is False


# ---------- QA ----------

def test_validate_numeric():
    assert validate_numeric(5) is True
    assert validate_numeric("5") is True
    assert validate_numeric(None) is False
    assert validate_numeric("abc") is False


def test_validate_percentage_bounds():
    assert validate_percentage(50) is True
    assert validate_percentage(999999) is False


def test_validate_kpis_flags_non_numeric():
    result = validate_kpis({"revenue": 100, "region": "Gauteng"})
    assert result["passed"] is False
    assert "region is not numeric." in result["errors"]


def test_build_and_validate_evidence():
    evidence = build_evidence(metric="total_revenue", value=1000)
    check = validate_evidence(evidence)
    assert check["passed"] is True


def test_validate_evidence_fails_on_missing_value():
    evidence = build_evidence(metric="total_revenue", value=None)
    check = validate_evidence(evidence)
    assert check["passed"] is False
    assert "value" in check["missing"]
