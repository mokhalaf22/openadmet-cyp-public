"""Invariants that must hold on the real dataset.

These require the pinned dataset in ./data (run `make data`). They skip cleanly
when it is absent so unit-test CI without the data still passes.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from cyp.guards import (
    assert_no_assigned_negatives,
    assigned_negative_mask,
    tdi_trainable_mask,
)
from cyp.losses import tdi_label_from_arms

TDI_FILE = Path("data/cyp-challenge-train-test/cyp-challenge-TRAIN_TDI.csv")
SCORED_ISOFORMS = ["CYP3A4", "CYP2D6"]

pytestmark = pytest.mark.skipif(
    not TDI_FILE.exists(), reason="dataset not downloaded (run `make data`)"
)


@pytest.fixture(scope="module")
def tdi():
    return pd.read_csv(TDI_FILE)


@pytest.mark.parametrize("iso", SCORED_ISOFORMS)
def test_derived_label_matches_truth_on_both_arms(tdi, iso):
    direct = tdi[f"{iso}_pIC50_direct_inhibition"]
    cond = tdi[f"{iso}_pIC50_TDI_condition"]
    truth = tdi[f"{iso}_is_TDI"]
    both = direct.notna() & cond.notna() & truth.notna()

    mu = torch.tensor(direct[both].to_numpy(), dtype=torch.float64)
    # delta = TDI_condition - direct (raw, so mu + delta == the measured TDI arm)
    delta = torch.tensor((cond - direct)[both].to_numpy(), dtype=torch.float64)
    derived = tdi_label_from_arms(mu, delta, hard=True).numpy()
    actual = truth[both].astype(bool).to_numpy()

    assert (derived == actual).all(), (
        f"{iso}: derived TDI label disagrees with is_TDI on "
        f"{(derived != actual).sum()} of {both.sum()} both-arms rows"
    )


@pytest.mark.parametrize("iso", SCORED_ISOFORMS)
def test_assigned_negatives_are_all_false_and_direct_absent(tdi, iso):
    mask = assigned_negative_mask(tdi, iso)
    if mask.sum() == 0:
        pytest.skip(f"no direct-less rows for {iso}")
    # Confirmed convention: these carry a default is_TDI=False and no direct arm.
    assert tdi.loc[mask, f"{iso}_is_TDI"].astype(bool).eq(False).all()
    assert tdi.loc[mask, f"{iso}_pIC50_direct_inhibition"].isna().all()


@pytest.mark.parametrize("iso", SCORED_ISOFORMS)
def test_guard_rejects_assigned_negatives_in_training(tdi, iso):
    assigned = assigned_negative_mask(tdi, iso)
    # The clean trainable selection must pass the guard...
    assert_no_assigned_negatives(tdi, iso, tdi_trainable_mask(tdi, iso))
    # ...but including any assigned-negative row must fail loudly.
    if assigned.sum() > 0:
        contaminated = tdi_trainable_mask(tdi, iso) | assigned
        with pytest.raises(ValueError, match="assigned-negative"):
            assert_no_assigned_negatives(tdi, iso, contaminated)


def test_trainable_and_assigned_negative_are_disjoint(tdi):
    for iso in SCORED_ISOFORMS:
        overlap = (tdi_trainable_mask(tdi, iso) & assigned_negative_mask(tdi, iso)).sum()
        assert overlap == 0
