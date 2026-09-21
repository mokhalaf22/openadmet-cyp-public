"""Validate and write challenge submission files (``make submit``).

Two independent tracks, each a separate file (`.parquet` preferred, `.csv`
accepted), exactly 750 rows. The schema is recorded verbatim from the Space
`config.py` (see CLAUDE.md "Submission schema"). The validator refuses to write
unless every check passes — a bad submission should fail loudly here, not on the
leaderboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- Schema, verbatim from the Space config.py (do not infer) ---------------
IDENTIFIER_COLUMNS = ["SMILES", "Molecule_Name"]
REGRESSION_ENDPOINTS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]
CLASSIFICATION_ENDPOINTS = ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]
REQUIRED_REGRESSION_COLUMNS = IDENTIFIER_COLUMNS + REGRESSION_ENDPOINTS
REQUIRED_CLASSIFICATION_COLUMNS = IDENTIFIER_COLUMNS + CLASSIFICATION_ENDPOINTS

N_ROWS = 750
PLAUSIBLE_PIC50 = (1.0, 10.0)

DATA_DIR = Path("data/cyp-challenge-train-test")
TEST_FILE = DATA_DIR / "cyp-challenge-TEST-BLINDED.csv"
BASELINE_PRED_FILE = Path("data/baseline_test_predictions.csv")
OUT_DIR = Path("submissions")

_TRACKS = {
    "regression": {
        "required": REQUIRED_REGRESSION_COLUMNS,
        "endpoints": REGRESSION_ENDPOINTS,
        "kind": "float",
    },
    "classification": {
        "required": REQUIRED_CLASSIFICATION_COLUMNS,
        "endpoints": CLASSIFICATION_ENDPOINTS,
        "kind": "bool",
    },
}


class SubmissionError(ValueError):
    """Raised when a predictions frame fails a submission check."""


def load_blinded_ids(path: Path = TEST_FILE) -> set:
    return set(pd.read_csv(path)["Molecule_Name"])


def validate_submission(df: pd.DataFrame, track: str, blinded_ids: set) -> None:
    """Raise SubmissionError on the first failing check; return None if valid."""
    if track not in _TRACKS:
        raise SubmissionError(f"unknown track {track!r}; expected one of {list(_TRACKS)}")
    spec = _TRACKS[track]
    required, endpoints = spec["required"], spec["endpoints"]

    # 1. Column names match the schema verbatim (no missing, no extras).
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SubmissionError(f"[{track}] missing required columns: {missing}")
    extra = [c for c in df.columns if c not in required]
    if extra:
        raise SubmissionError(f"[{track}] unexpected columns not in schema: {extra}")

    # 2. Exactly 750 rows.
    if len(df) != N_ROWS:
        raise SubmissionError(f"[{track}] expected {N_ROWS} rows, got {len(df)}")

    # 3. Molecule_Name set matches the blinded file exactly (order-independent).
    if df["Molecule_Name"].duplicated().any():
        n = int(df["Molecule_Name"].duplicated().sum())
        raise SubmissionError(f"[{track}] {n} duplicate Molecule_Name rows")
    got = set(df["Molecule_Name"])
    if got != blinded_ids:
        missing_ids = blinded_ids - got
        extra_ids = got - blinded_ids
        raise SubmissionError(
            f"[{track}] Molecule_Name set does not match the blinded file: "
            f"{len(missing_ids)} missing, {len(extra_ids)} unexpected"
        )

    # 4. No NaN in any prediction column.
    for col in endpoints:
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise SubmissionError(f"[{track}] prediction column {col} contains {n} NaN(s)")

    # 5. Per-kind dtype / value checks.
    if spec["kind"] == "bool":
        for col in endpoints:
            if df[col].dtype != np.dtype("bool"):
                raise SubmissionError(
                    f"[{track}] TDI column {col} must be boolean dtype, "
                    f"got {df[col].dtype} (probabilities/objects are rejected)"
                )
    else:
        for col in endpoints:
            vals = df[col].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise SubmissionError(f"[{track}] prediction column {col} contains inf")
            lo, hi = PLAUSIBLE_PIC50
            bad = (vals < lo) | (vals > hi)
            if bad.any():
                raise SubmissionError(
                    f"[{track}] pIC50 column {col} has {int(bad.sum())} value(s) "
                    f"outside the plausible range [{lo}, {hi}] "
                    f"(e.g. {np.round(vals[bad][:3], 3).tolist()})"
                )


def write_submission(df: pd.DataFrame, track: str, out_path, blinded_ids=None) -> Path:
    """Validate then write. Refuses to write anything that fails validation."""
    if blinded_ids is None:
        blinded_ids = load_blinded_ids()
    validate_submission(df, track, blinded_ids)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise SubmissionError(f"output must be .parquet or .csv, got {out_path.suffix!r}")
    return out_path


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a bool/int/'True'/'False' column to a genuine bool dtype."""
    if series.dtype == np.dtype("bool"):
        return series
    mapping = {True: True, False: False, "True": True, "False": False,
               "true": True, "false": False, 1: True, 0: False}
    return series.map(mapping).astype("bool")


def map_baseline_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map baseline internal names ({iso}_pIC50, {iso}_is_TDI) to the official
    per-track submission frames."""
    reg = df[IDENTIFIER_COLUMNS].copy()
    for endpoint in REGRESSION_ENDPOINTS:
        iso = endpoint.split("_", 1)[0]
        reg[endpoint] = df[f"{iso}_pIC50"].astype(float)
    reg = reg[REQUIRED_REGRESSION_COLUMNS]

    cls = df[IDENTIFIER_COLUMNS].copy()
    for endpoint in CLASSIFICATION_ENDPOINTS:
        iso = endpoint.split("_", 1)[0]
        cls[endpoint] = _to_bool(df[f"{iso}_is_TDI"])
    cls = cls[REQUIRED_CLASSIFICATION_COLUMNS]
    return reg, cls


def main() -> int:
    if not BASELINE_PRED_FILE.exists():
        print(f"Missing {BASELINE_PRED_FILE}. Run `make baseline` first.")
        return 1
    blinded_ids = load_blinded_ids()
    preds = pd.read_csv(BASELINE_PRED_FILE)
    reg, cls = map_baseline_predictions(preds)

    reg_path = write_submission(reg, "regression", OUT_DIR / "regression.parquet", blinded_ids)
    cls_path = write_submission(cls, "classification", OUT_DIR / "classification.parquet", blinded_ids)
    print(f"validated + wrote:\n  {reg_path}  ({reg.shape[0]} rows, {reg.shape[1]} cols)")
    print(f"  {cls_path}  ({cls.shape[0]} rows, {cls.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
