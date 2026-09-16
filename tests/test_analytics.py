import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_sales_file
from src.analytics.kpis import calculate_core_kpis, calculate_profitability_kpis
from src.analytics.targets import calculate_target_performance
from src.analytics.products import product_performance, product_growth, category_performance
from src.analytics.customers import customer_performance, rfm_analysis
from src.analytics.stores import store_performance, regional_performance
from src.analytics.sales import monthly_sales, calculate_monthly_growth, calculate_yoy_growth, daily_sales
from src.analytics.pareto import pareto_analysis
from src.analytics.promotions import promotion_performance
from src.analytics.anomalies import detect_anomalies, rolling_sales_anomalies
from src.analytics.engine import run_analytics

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "fmcg_sales_sample.csv"


def _sample_df():
    return load_sales_file(SAMPLE)


# ---------- KPIs ----------

def test_core_kpis_basic():
    df = pd.DataFrame({
        "transaction_id": ["TX1", "TX2", "TX3"],
        "customer": ["A", "B", "A"],
        "quantity": [10, 20, 5],
        "revenue": [100, 200, 50],
    })
    result = calculate_core_kpis(df)
    assert result["total_revenue"] == 350
    assert result["total_units"] == 35
    assert result["transactions"] == 3
    assert result["customers"] == 2
    assert round(result["average_order_value"], 2) == round(350 / 3, 2)


def test_profitability_kpis():
    df = pd.DataFrame({"revenue": [100, 200], "cost": [60, 100], "quantity": [10, 20]})
    result = calculate_profitability_kpis(df)
    assert result["revenue"] == 300
    assert result["cogs"] == 160
    assert result["gross_profit"] == 140
    assert round(result["gross_margin_pct"], 2) == round(140 / 300 * 100, 2)


def test_profitability_kpis_missing_cost_returns_empty():
    df = pd.DataFrame({"revenue": [100, 200]})
    assert calculate_profitability_kpis(df) == {}


def test_kpis_on_real_sample_are_sane():
    df = _sample_df()
    kpis = calculate_core_kpis(df)
    assert kpis["total_revenue"] > 0
    assert kpis["total_units"] > 0
    assert kpis["transactions"] == df["transaction_id"].nunique()


# ---------- Targets ----------

def test_target_performance_available():
    df = pd.DataFrame({"revenue": [100, 200], "target": [90, 220]})
    result = calculate_target_performance(df)
    assert result["available"] is True
    assert result["actual"] == 300
    assert result["target"] == 310
    assert round(result["achievement_pct"], 2) == round(300 / 310 * 100, 2)


def test_target_performance_unavailable_without_target_col():
    df = pd.DataFrame({"revenue": [100, 200]})
    result = calculate_target_performance(df)
    assert result["available"] is False


# ---------- Products / categories ----------

def test_product_performance_sums_and_sorts():
    df = pd.DataFrame({
        "product": ["A", "B", "A", "C"],
        "revenue": [100, 500, 50, 10],
        "quantity": [1, 2, 1, 1],
    })
    result = product_performance(df)
    assert result.iloc[0]["product"] == "B"
    assert result[result["product"] == "A"]["revenue"].iloc[0] == 150


def test_product_growth_computes_mom_pct_change():
    df = pd.DataFrame({
        "product": ["A", "A"],
        "date": ["2026-01-01", "2026-02-01"],
        "revenue": [100, 150],
    })
    result = product_growth(df)
    assert result.iloc[1]["growth_pct"] == 50.0


def test_category_performance_on_real_sample():
    df = _sample_df()
    result = category_performance(df)
    assert result is not None
    assert set(result["category"]) == {"RTE", "Processed", "Fresh"}


# ---------- Customers ----------

def test_customer_performance_none_without_customer_column():
    df = pd.DataFrame({"revenue": [100, 200], "product": ["A", "B"]})
    assert customer_performance(df) is None


def test_rfm_analysis_basic():
    df = pd.DataFrame({
        "customer": ["A", "A", "B"],
        "date": ["2026-01-01", "2026-01-10", "2026-01-05"],
        "revenue": [100, 100, 50],
        "transaction_id": ["T1", "T2", "T3"],
    })
    result = rfm_analysis(df)
    assert result is not None
    a_row = result[result["customer"] == "A"].iloc[0]
    assert a_row["frequency"] == 2
    assert a_row["monetary"] == 200


# ---------- Stores / regions ----------

def test_store_and_region_performance_on_real_sample():
    df = _sample_df()
    stores = store_performance(df)
    regions = regional_performance(df)
    assert set(stores["store"]) == {"Polokwane", "Pretoria", "Johannesburg", "Nelspruit"}
    assert set(regions["region"]) == {"Limpopo", "Gauteng", "Mpumalanga"}


# ---------- Sales trends ----------

def test_monthly_sales_aggregates_correctly():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-15", "2026-02-01"]),
        "revenue": [100, 200, 300],
        "quantity": [10, 20, 30],
    })
    result = monthly_sales(df)
    assert len(result) == 2
    assert result.iloc[0]["revenue"] == 300
    assert result.iloc[1]["revenue"] == 300


def test_monthly_growth_pct():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-02-01"]), "revenue": [100, 200]})
    result = calculate_monthly_growth(df)
    assert result.iloc[1]["revenue_mom_growth_pct"] == 100


def test_yoy_growth_needs_12_months_lookback():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=13, freq="MS"),
        "revenue": [100] * 12 + [150],
    })
    result = calculate_yoy_growth(df)
    assert result.iloc[-1]["revenue_yoy_growth_pct"] == 50.0


def test_daily_sales_on_real_sample_matches_date_range():
    df = _sample_df()
    daily = daily_sales(df)
    assert daily["period"].min() == pd.Timestamp("2025-01-01")
    assert daily["period"].max() == pd.Timestamp("2026-08-31")


# ---------- Pareto ----------

def test_pareto_analysis_cumulative_and_flag():
    df = pd.DataFrame({"product": ["A", "B", "C"], "revenue": [800, 150, 50]})
    result = pareto_analysis(df, "product")
    assert result.iloc[0]["product"] == "A"
    assert result.iloc[0]["cumulative_contribution_pct"] == 80.0
    assert bool(result.iloc[0]["pareto_80"]) is True


def test_pareto_on_real_sample_identifies_top_products():
    df = _sample_df()
    result = pareto_analysis(df, "product")
    assert result is not None
    assert result.iloc[0]["revenue"] >= result.iloc[-1]["revenue"]


# ---------- Promotions ----------

def test_promotion_performance_unavailable_without_flag():
    df = pd.DataFrame({"revenue": [100, 200]})
    result = promotion_performance(df)
    assert result["available"] is False


def test_promotion_performance_segments_correctly():
    df = pd.DataFrame({
        "revenue": [100, 200, 50],
        "quantity": [1, 2, 1],
        "promotion": ["yes", "no", "yes"],
    })
    result = promotion_performance(df)
    assert result["available"] is True
    assert result["promoted"]["revenue"] == 150
    assert result["non_promoted"]["revenue"] == 200


# ---------- Anomalies ----------

def test_detect_anomalies_unavailable_with_short_history():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "revenue": [100, 110, 105],
    })
    result = detect_anomalies(df)
    assert result["available"] is False


def test_rolling_anomalies_flags_extreme_spike():
    dates = pd.date_range("2026-01-01", periods=30)
    revenue = [1000] * 29 + [50000]  # obvious spike on the last day
    df = pd.DataFrame({"date": dates, "revenue": revenue})
    result = rolling_sales_anomalies(df, window=7, threshold=2)
    assert result is not None
    assert bool(result.iloc[-1]["is_anomaly"]) is True


# ---------- Engine ----------

def test_run_analytics_end_to_end_on_real_sample():
    df = _sample_df()
    results = run_analytics(df)
    assert results["core_kpis"]["total_revenue"] > 0
    assert results["target_performance"]["available"] is True
    assert results["profitability"]  # cost column exists in sample
    assert results["products"] is not None
    assert results["pareto_products"] is not None
    assert results["anomalies"]["available"] is True
