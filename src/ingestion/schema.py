"""
Schema detection: map whatever columns a source system used onto the
canonical roles our analytics engine expects (revenue, quantity, date, etc).
This is what lets the system work across different FMCG data exports
without hardcoding column names.
"""
from __future__ import annotations
import pandas as pd

# Canonical role -> list of column-name candidates (already lower/underscored
# by loader.load_sales_file before this runs).
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "revenue": ["revenue", "sales", "sales_value", "turnover", "net_sales"],
    "quantity": ["quantity", "qty", "units", "volume", "mass"],
    "cost": ["cost", "cogs", "cost_of_goods_sold", "cost_of_sales"],
    "date": ["date", "sales_date", "transaction_date", "order_date", "invoice_date"],
    "transaction_id": ["transaction_id", "transaction", "invoice_id", "invoice", "order_id", "order"],
    "customer": ["customer", "customer_id", "customer_name", "client", "client_id", "debtor", "debtor_code"],
    "product": ["product", "product_name", "sku", "sku_name", "product_id"],
    "category": ["category", "product_category", "department", "category_name"],
    "store": ["store", "store_name", "branch", "branch_name", "store_id"],
    "region": ["region", "province", "area", "territory", "region_name"],
    "target": ["target", "sales_target", "budget", "sales_budget"],
    "promotion": ["promotion", "promo", "promo_flag", "on_promotion", "campaign"],
    "price": ["price", "unit_price", "selling_price"],
    "discount": ["discount", "discount_value", "discount_amount"],
}


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """First matching column name from a candidate list (case already normalized)."""
    normalized = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def detect_schema(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Detect which canonical roles are present in this dataset and which
    actual column fills each role. Missing roles map to None so downstream
    analytics modules can gracefully skip calculations they can't support.
    """
    return {
        role: find_column(df, candidates)
        for role, candidates in COLUMN_CANDIDATES.items()
    }


def schema_summary(schema: dict[str, str | None]) -> dict[str, list[str]]:
    """Split a detected schema into what's available vs missing, for display/QA."""
    available = {role: col for role, col in schema.items() if col}
    missing = [role for role, col in schema.items() if not col]
    return {"available": available, "missing_roles": missing}
