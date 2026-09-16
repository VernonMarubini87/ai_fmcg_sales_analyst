"""
Phase 3 — Core FMCG KPI engine. Deterministic only; no AI involved here.
Schema-adaptive: only calculates a KPI when the underlying columns exist.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from src.ingestion.schema import detect_schema


def calculate_core_kpis(df: pd.DataFrame) -> dict:
    schema = detect_schema(df)
    kpis = {}

    rev_col = schema.get("revenue")
    qty_col = schema.get("quantity")
    tx_col = schema.get("transaction_id")

    if rev_col:
        kpis["total_revenue"] = float(pd.to_numeric(df[rev_col], errors="coerce").sum())

    if qty_col:
        kpis["total_units"] = float(pd.to_numeric(df[qty_col], errors="coerce").sum())

    if tx_col:
        kpis["transactions"] = int(df[tx_col].nunique())
    else:
        kpis["transactions"] = len(df)

    customer_col = schema.get("customer")
    if customer_col:
        kpis["customers"] = int(df[customer_col].nunique())

    if "total_revenue" in kpis and kpis["transactions"] > 0:
        kpis["average_order_value"] = kpis["total_revenue"] / kpis["transactions"]

    if "total_revenue" in kpis and "total_units" in kpis and kpis["total_units"] != 0:
        kpis["average_selling_price"] = kpis["total_revenue"] / kpis["total_units"]

    if "total_revenue" in kpis and "customers" in kpis and kpis["customers"] > 0:
        kpis["revenue_per_customer"] = kpis["total_revenue"] / kpis["customers"]

    if "total_units" in kpis and kpis["transactions"] > 0:
        kpis["units_per_transaction"] = kpis["total_units"] / kpis["transactions"]

    return kpis


def calculate_profitability_kpis(df: pd.DataFrame) -> dict:
    schema = detect_schema(df)
    kpis = {}

    rev_col, cost_col, qty_col = schema.get("revenue"), schema.get("cost"), schema.get("quantity")

    if not (rev_col and cost_col):
        return kpis

    revenue = float(pd.to_numeric(df[rev_col], errors="coerce").sum())
    cost = float(pd.to_numeric(df[cost_col], errors="coerce").sum())
    gross_profit = revenue - cost

    kpis["revenue"] = revenue
    kpis["cogs"] = cost
    kpis["gross_profit"] = gross_profit

    if revenue != 0:
        kpis["gross_margin_pct"] = gross_profit / revenue * 100

    if qty_col:
        total_units = float(pd.to_numeric(df[qty_col], errors="coerce").sum())
        if total_units != 0:
            kpis["profit_per_unit"] = gross_profit / total_units

    return kpis
