"""Print the schema of every downloaded data file.

Run this FIRST (``make inspect``). It walks ``./data`` for CSV and parquet
files and, for each one, prints the shape and a per-column summary: name,
dtype, non-null count, and — for numeric columns — min / median / max.

The point is to confirm the real column names and value ranges before any other
code in this package refers to them. Nothing here hardcodes a column name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

DATA_DIR = Path("data")


def find_data_files(root: Path) -> list[Path]:
    """Return CSV/parquet files under ``root``, sorted, skipping HF caches."""
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".csv", ".parquet"}
        and ".cache" not in p.parts
    ]
    return sorted(files)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def describe_column(series: pd.Series) -> str:
    """One-line summary for a single column."""
    non_null = int(series.notna().sum())
    line = f"non_null={non_null}/{len(series)}"
    if ptypes.is_numeric_dtype(series) and non_null > 0:
        line += (
            f"  min={series.min():.4g}"
            f"  median={series.median():.4g}"
            f"  max={series.max():.4g}"
        )
    return line


def describe_file(path: Path) -> None:
    print("=" * 78)
    print(f"FILE: {path}")
    try:
        df = read_table(path)
    except Exception as exc:  # noqa: BLE001 — inspection should not abort on one bad file
        print(f"  !! could not read: {type(exc).__name__}: {exc}")
        return

    n_rows, n_cols = df.shape
    print(f"SHAPE: {n_rows} rows x {n_cols} cols")
    name_width = max((len(str(c)) for c in df.columns), default=0)
    for col in df.columns:
        summary = describe_column(df[col])
        print(f"  {str(col):<{name_width}}  {str(df[col].dtype):<10}  {summary}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]) if argv else DATA_DIR
    if not root.exists():
        print(f"No data directory at {root!r}. Download the dataset first.")
        return 1

    files = find_data_files(root)
    if not files:
        print(f"No CSV/parquet files found under {root!r}.")
        return 1

    print(f"Found {len(files)} data file(s) under {root}/\n")
    for path in files:
        describe_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
