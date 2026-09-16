from __future__ import annotations
import pandas as pd
from src.ingestion.schema import detect_schema


def promotion_performance(df: pd.DataFrame) -> dict:
    """
    Compares promoted vs non-promoted revenue/units/avg price, when a
    promotion flag column exists. Returns available=False otherwise
    rather than guessing.
    """
    schema = detect_schema(df)
    promo_col, rev_col, qty_col = schema.get("promotion"), schema.get("revenue"), schema.get("quantity")

    if not promo_col or not rev_col:
        return {"available": False, "reason": "No promotion flag column detected."}

    temp = df.copy()
    promo_flag = temp[promo_col].astype(str).str.lower().isin(["1", "true", "yes", "y", "promo"])

    def _segment_stats(mask):
        seg_rev = float(pd.to_numeric(temp.loc[mask, rev_col], errors="coerce").sum())
        seg_units = float(pd.to_numeric(temp.loc[mask, qty_col], errors="coerce").sum()) if qty_col else None
        stats = {"revenue": seg_rev, "transactions": int(mask.sum())}
        if seg_units is not None:
            stats["units"] = seg_units
            stats["average_price"] = (seg_rev / seg_units) if seg_units else None
        return stats

    return {
        "available": True,
        "promoted": _segment_stats(promo_flag),
        "non_promoted": _segment_stats(~promo_flag),
    }
