from __future__ import annotations
import pandas as pd

from src.analytics.kpis import calculate_core_kpis, calculate_profitability_kpis
from src.analytics.targets import calculate_target_performance
from src.analytics.products import product_performance, category_performance, product_growth
from src.analytics.customers import customer_performance, rfm_analysis
from src.analytics.stores import store_performance, regional_performance
from src.analytics.sales import (
    daily_sales, weekly_sales, monthly_sales,
    calculate_monthly_growth, calculate_yoy_growth, volume_price_analysis,
)
from src.analytics.pareto import pareto_analysis
from src.analytics.promotions import promotion_performance
from src.analytics.anomalies import detect_anomalies


def run_analytics(df: pd.DataFrame) -> dict:
    """
    Single entry point: runs every analytics module against the cleaned
    dataframe and returns a results dict. Every module is schema-adaptive,
    so results contain None / available=False where the source data
    doesn't support that analysis, rather than raising.
    """
    return {
        "core_kpis": calculate_core_kpis(df),
        "profitability": calculate_profitability_kpis(df),
        "target_performance": calculate_target_performance(df),

        "daily_sales": daily_sales(df),
        "weekly_sales": weekly_sales(df),
        "monthly_sales": monthly_sales(df),
        "monthly_growth": calculate_monthly_growth(df),
        "yoy_growth": calculate_yoy_growth(df),
        "volume_price": volume_price_analysis(df),

        "products": product_performance(df),
        "product_growth": product_growth(df),
        "categories": category_performance(df),

        "customers": customer_performance(df),
        "rfm": rfm_analysis(df),

        "stores": store_performance(df),
        "regions": regional_performance(df),

        "pareto_products": pareto_analysis(df, "product"),
        "pareto_customers": pareto_analysis(df, "customer"),
        "pareto_stores": pareto_analysis(df, "store"),

        "promotions": promotion_performance(df),
        "anomalies": detect_anomalies(df),
    }
