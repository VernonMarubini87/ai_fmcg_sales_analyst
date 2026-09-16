from __future__ import annotations


def validate_numeric(value) -> bool:
    if value is None:
        return False
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def validate_percentage(value) -> bool:
    if not validate_numeric(value):
        return False
    return -10000 <= float(value) <= 10000


def validate_kpis(kpis: dict) -> dict:
    """Checks that every KPI value is numeric before it's allowed to reach Claude or the UI."""
    errors = []
    for key, value in kpis.items():
        if not validate_numeric(value):
            errors.append(f"{key} is not numeric.")
    return {"passed": len(errors) == 0, "errors": errors}
