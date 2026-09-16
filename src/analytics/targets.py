from __future__ import annotations
import pandas as pd
from src.ingestion.schema import detect_schema


def calculate_target_performance(df: pd.DataFrame) -> dict:
    schema = detect_schema(df)
    rev_col, target_col = schema.get("revenue"), schema.get("target")

    if not rev_col or not target_col:
        return {"available": False, "reason": "Revenue and/or target columns were not found."}

    actual = float(pd.to_numeric(df[rev_col], errors="coerce").sum())
    target = float(pd.to_numeric(df[target_col], errors="coerce").sum())
    variance = actual - target

    variance_pct = (variance / target * 100) if target != 0 else None
    achievement_pct = (actual / target * 100) if target != 0 else None

    return {
        "available": True,
        "actual": actual,
        "target": target,
        "variance": variance,
        "variance_pct": variance_pct,
        "achievement_pct": achievement_pct,
    }
