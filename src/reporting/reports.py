from __future__ import annotations
import pandas as pd
from datetime import date


def generate_management_report(analytics: dict) -> str:
    """
    Produces a deterministic, evidence-linked text report from analytics
    results — no Claude required for this baseline version, so a report
    can always be generated even without an API key configured.
    """
    lines = [f"# FMCG Sales Management Report", f"_Generated {date.today().isoformat()}_", ""]

    core = analytics.get("core_kpis", {})
    lines.append("## Executive Summary")
    if "total_revenue" in core:
        lines.append(f"- Total revenue: R{core['total_revenue']:,.2f}")
    if "total_units" in core:
        lines.append(f"- Total units: {core['total_units']:,.0f}")
    if "transactions" in core:
        lines.append(f"- Transactions: {core['transactions']:,}")
    if "average_order_value" in core:
        lines.append(f"- Average order value: R{core['average_order_value']:,.2f}")

    profitability = analytics.get("profitability", {})
    if profitability:
        lines.append("")
        lines.append("## Profitability")
        lines.append(f"- Gross profit: R{profitability.get('gross_profit', 0):,.2f}")
        if "gross_margin_pct" in profitability:
            lines.append(f"- Gross margin: {profitability['gross_margin_pct']:.1f}%")
    else:
        lines.append("")
        lines.append("## Profitability")
        lines.append("- Not available: no cost/COGS column detected in the source data.")

    target = analytics.get("target_performance", {})
    lines.append("")
    lines.append("## Target Performance")
    if target.get("available"):
        lines.append(f"- Actual: R{target['actual']:,.2f} vs Target: R{target['target']:,.2f}")
        lines.append(f"- Achievement: {target['achievement_pct']:.1f}%")
    else:
        lines.append(f"- Not available: {target.get('reason', 'no target data.')}")

    products = analytics.get("products")
    if isinstance(products, pd.DataFrame) and not products.empty:
        lines.append("")
        lines.append("## Top Products by Revenue")
        for _, row in products.head(5).iterrows():
            name = row.iloc[0]
            lines.append(f"- {name}: R{row['revenue']:,.2f} ({row.get('revenue_contribution_pct', 0):.1f}% of revenue)")

    anomalies = analytics.get("anomalies", {})
    lines.append("")
    lines.append("## Anomalies")
    if anomalies.get("available"):
        lines.append(f"- {anomalies['anomaly_count']} unusual daily-revenue observation(s) flagged (rolling 7-day, 2σ threshold).")
    else:
        lines.append(f"- Not available: {anomalies.get('reason', 'insufficient history.')}")

    lines.append("")
    lines.append("## Data Limitations")
    lines.append("- This report only states what the underlying data supports. "
                  "Any metric marked 'not available' means the source file lacked the required column(s).")

    return "\n".join(lines)
