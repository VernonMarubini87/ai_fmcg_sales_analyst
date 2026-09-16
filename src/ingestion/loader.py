"""
Data ingestion: load CSV/Excel sales exports into a clean pandas DataFrame.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path


class UnsupportedFileType(Exception):
    pass


def load_sales_file(file_path: str | Path) -> pd.DataFrame:
    """
    Load a sales data file (.csv, .xlsx, .xls) into a DataFrame.
    Column names are stripped and lower-cased with spaces -> underscores
    so downstream `find_column` matching is reliable regardless of how
    the source system named its fields.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise UnsupportedFileType(
            f"Unsupported file type '{suffix}'. Use .csv, .xlsx, or .xls."
        )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    return df
