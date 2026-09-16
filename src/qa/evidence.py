from __future__ import annotations


def build_evidence(metric: str, value, comparison=None, period=None) -> dict:
    return {"metric": metric, "value": value, "comparison": comparison, "period": period}


def validate_evidence(evidence: dict) -> dict:
    """
    Evidence must at minimum name what it's about and carry a non-null,
    non-empty value. This is the last gate before evidence is either
    shown directly or handed to Claude for interpretation.
    """
    missing = []
    if not evidence.get("metric"):
        missing.append("metric")

    value = evidence.get("value")
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        missing.append("value")

    return {"passed": len(missing) == 0, "missing": missing}
