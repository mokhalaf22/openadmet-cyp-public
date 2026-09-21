"""Every submission check must trip on a deliberately broken frame."""

import numpy as np
import pandas as pd
import pytest

from cyp import submit
from cyp.submit import (
    REQUIRED_CLASSIFICATION_COLUMNS,
    REQUIRED_REGRESSION_COLUMNS,
    SubmissionError,
    map_baseline_predictions,
    validate_submission,
)

IDS = [f"MOL-{i:04d}" for i in range(submit.N_ROWS)]
BLINDED = set(IDS)


def good_regression():
    df = pd.DataFrame({"SMILES": ["CCO"] * submit.N_ROWS, "Molecule_Name": IDS})
    for col in submit.REGRESSION_ENDPOINTS:
        df[col] = 5.0
    return df[REQUIRED_REGRESSION_COLUMNS]


def good_classification():
    df = pd.DataFrame({"SMILES": ["CCO"] * submit.N_ROWS, "Molecule_Name": IDS})
    for col in submit.CLASSIFICATION_ENDPOINTS:
        df[col] = np.zeros(submit.N_ROWS, dtype=bool)
    return df[REQUIRED_CLASSIFICATION_COLUMNS]


def test_valid_frames_pass():
    validate_submission(good_regression(), "regression", BLINDED)
    validate_submission(good_classification(), "classification", BLINDED)


def test_missing_column_trips():
    df = good_regression().drop(columns=["CYP3A4_pIC50_direct_inhibition"])
    with pytest.raises(SubmissionError, match="missing required columns"):
        validate_submission(df, "regression", BLINDED)


def test_extra_column_trips():
    df = good_regression().copy()
    df["surprise"] = 1.0
    with pytest.raises(SubmissionError, match="unexpected columns"):
        validate_submission(df, "regression", BLINDED)


def test_wrong_row_count_trips():
    df = good_regression().iloc[:-1]
    with pytest.raises(SubmissionError, match="expected 750 rows"):
        validate_submission(df, "regression", BLINDED)


def test_id_set_mismatch_trips():
    df = good_regression().copy()
    df.loc[0, "Molecule_Name"] = "NOT-A-REAL-ID"
    with pytest.raises(SubmissionError, match="does not match the blinded file"):
        validate_submission(df, "regression", BLINDED)


def test_duplicate_ids_trip():
    df = good_regression().copy()
    df.loc[1, "Molecule_Name"] = df.loc[0, "Molecule_Name"]
    with pytest.raises(SubmissionError, match="duplicate Molecule_Name"):
        validate_submission(df, "regression", BLINDED)


def test_nan_prediction_trips():
    df = good_regression().copy()
    df.loc[0, "CYP1A2_pIC50_direct_inhibition"] = np.nan
    with pytest.raises(SubmissionError, match="contains .*NaN"):
        validate_submission(df, "regression", BLINDED)


def test_pic50_out_of_range_trips():
    df = good_regression().copy()
    df.loc[0, "CYP2D6_pIC50_direct_inhibition"] = 15.0
    with pytest.raises(SubmissionError, match="outside the plausible range"):
        validate_submission(df, "regression", BLINDED)


def test_inf_prediction_trips():
    df = good_regression().copy()
    df.loc[0, "CYP2C9_pIC50_direct_inhibition"] = np.inf
    with pytest.raises(SubmissionError, match="contains inf"):
        validate_submission(df, "regression", BLINDED)


def test_non_bool_tdi_trips():
    df = good_classification().copy()
    df["CYP3A4_is_TDI"] = 0.7  # probabilities, not booleans
    with pytest.raises(SubmissionError, match="must be boolean dtype"):
        validate_submission(df, "classification", BLINDED)


def test_object_bool_tdi_trips():
    df = good_classification().copy()
    df["CYP2D6_is_TDI"] = ["True"] * submit.N_ROWS  # strings, object dtype
    with pytest.raises(SubmissionError, match="must be boolean dtype"):
        validate_submission(df, "classification", BLINDED)


def test_write_refuses_invalid(tmp_path):
    df = good_regression().iloc[:-1]
    with pytest.raises(SubmissionError):
        submit.write_submission(df, "regression", tmp_path / "r.parquet", BLINDED)
    assert not (tmp_path / "r.parquet").exists()


def test_map_baseline_predictions_produces_official_schema():
    base = pd.DataFrame({"SMILES": ["CCO"] * submit.N_ROWS, "Molecule_Name": IDS})
    for iso in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
        base[f"{iso}_pIC50"] = 5.0
    for iso in ["CYP2D6", "CYP3A4"]:
        base[f"{iso}_is_TDI"] = np.ones(submit.N_ROWS, dtype=bool)
    reg, cls = map_baseline_predictions(base)
    assert list(reg.columns) == REQUIRED_REGRESSION_COLUMNS
    assert list(cls.columns) == REQUIRED_CLASSIFICATION_COLUMNS
    assert cls["CYP3A4_is_TDI"].dtype == np.dtype("bool")
    validate_submission(reg, "regression", BLINDED)
    validate_submission(cls, "classification", BLINDED)
