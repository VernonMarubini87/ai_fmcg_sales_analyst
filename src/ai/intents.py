from __future__ import annotations

# Maps a user-facing intent to the corresponding key in run_analytics()'s output.
INTENT_TO_RESULT_KEY = {
    "sales_summary": "core_kpis",
    "profitability": "profitability",
    "sales_trend": "monthly_sales",
    "growth": "monthly_growth",
    "yoy_growth": "yoy_growth",
    "product_analysis": "products",
    "product_growth": "product_growth",
    "category_analysis": "categories",
    "customer_analysis": "customers",
    "rfm": "rfm",
    "store_analysis": "stores",
    "regional_analysis": "regions",
    "target_analysis": "target_performance",
    "pareto_products": "pareto_products",
    "pareto_customers": "pareto_customers",
    "promotion_analysis": "promotions",
    "anomaly_analysis": "anomalies",
    "volume_price": "volume_price",
}

INTENT_KEYWORDS = {
    "sales_summary": ["total revenue", "overall", "summary", "how are we doing"],
    "sales_trend": ["trend", "over time", "monthly"],
    "growth": ["growth", "grow", "increase", "decrease", "decline", "mom", "month on month"],
    "yoy_growth": ["year on year", "yoy", "vs last year"],
    "product_analysis": ["product", "sku"],
    "customer_analysis": ["customer", "client"],
    "store_analysis": ["store", "branch"],
    "regional_analysis": ["region", "province"],
    "target_analysis": ["target", "budget", "achievement"],
    "pareto_products": ["80/20", "pareto", "top products"],
    "promotion_analysis": ["promotion", "promo"],
    "anomaly_analysis": ["anomaly", "unusual", "spike", "drop"],
    "rfm": ["rfm", "recency", "frequency", "high value customers"],
}


def classify_intent(question: str) -> str:
    """
    Lightweight keyword-based intent classifier. Deliberately simple and
    deterministic rather than an LLM call, so routing is fast, free, and
    auditable — Claude is reserved for interpretation, not routing.
    """
    q = question.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "sales_summary"
