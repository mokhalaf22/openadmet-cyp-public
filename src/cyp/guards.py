"""Data guards for the TDI track.

These matter more than the loss: the single easiest way to score a bad TDI model
is to train or validate on rows whose `is_TDI` label is a default rather than a
measurement. This module makes that mistake fail loudly.
"""

from __future__ import annotations

import pandas as pd


def _arm_columns(isoform: str) -> tuple[str, str]:
    return (
        f"{isoform}_pIC50_direct_inhibition",
        f"{isoform}_pIC50_TDI_condition",
    )


def assigned_negative_mask(df: pd.DataFrame, isoform: str) -> pd.Series:
    """Rows where the TDI arm was measured but the direct arm was NOT assayed.

    The direct DRC is entirely absent (every direct-side column NaN; the Emax
    file confirms the direct arm was never run), so no shift could be computed
    and `is_TDI` is an assigned default — it carries no experimental information
    about the compound. These rows must never enter TDI classification training
    or internal evaluation, and `mu` must never be supervised on them.

    NOTE: this is DISTINCT from the organizers' formal "assigned negative" class
    (direct pIC50 < 4 AND TDI-arm pIC50 < 4, both arms measured and inactive),
    which is a scored class on the blinded test. See CLAUDE.md / FINDINGS.md.
    """
    direct, cond = _arm_columns(isoform)
    return df[cond].notna() & df[direct].isna()


def tdi_trainable_mask(df: pd.DataFrame, isoform: str) -> pd.Series:
    """Rows usable for TDI classification: both arms measured (so the label is a
    real, derivable shift) and `is_TDI` present."""
    direct, cond = _arm_columns(isoform)
    label = f"{isoform}_is_TDI"
    return df[direct].notna() & df[cond].notna() & df[label].notna()


def _selected_index(df: pd.DataFrame, rows) -> pd.Index:
    """Coerce a boolean mask or an index-like into a positional-safe Index."""
    if isinstance(rows, pd.Series) and rows.dtype == bool:
        return df.index[rows.to_numpy()]
    return pd.Index(rows)


def assert_no_assigned_negatives(df: pd.DataFrame, isoform: str, rows) -> None:
    """Fail loudly if any assigned-negative (direct-arm-not-assayed) row is in
    `rows` — the index or boolean mask selected for TDI train/eval.

    Call this at the point you build a TDI training or evaluation split, for
    every scored isoform (CYP3A4, CYP2D6).
    """
    bad = df.index[assigned_negative_mask(df, isoform).to_numpy()]
    offending = bad.intersection(_selected_index(df, rows))
    if len(offending) > 0:
        raise ValueError(
            f"{len(offending)} assigned-negative {isoform} rows (direct arm "
            f"never assayed; is_TDI is a default, not a measurement) are present "
            f"in the TDI train/eval selection and must be excluded. "
            f"Example indices: {list(offending[:5])}. "
            f"Use guards.tdi_trainable_mask to select rows instead."
        )
