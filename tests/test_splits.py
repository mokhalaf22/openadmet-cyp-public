"""Scaffold-split determinism and group integrity.

The baseline is the reference every later model is measured against, so the fold
assignment must be reproducible and must never leak a scaffold across the
train/val boundary.
"""

import numpy as np

from cyp.splits import scaffold_folds

# A small mixed set: several benzene analogs, several pyridine analogs, etc.
SMILES = [
    "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1O", "c1ccccc1N",
    "c1ccncc1C", "c1ccncc1CC", "c1ccncc1O", "c1ccncc1N",
    "C1CCCCC1C", "C1CCCCC1CC", "C1CCCCC1O",
    "c1ccc2ccccc2c1", "c1ccc2ccccc2c1C", "c1ccc2ccccc2c1O",
    "CCO", "CCCO", "CCCCO",  # acyclic -> empty scaffold, all one group
]


def test_scaffold_folds_deterministic():
    a, ga = scaffold_folds(SMILES, n_folds=3, seed=0)
    b, gb = scaffold_folds(SMILES, n_folds=3, seed=0)
    assert np.array_equal(a, b)
    assert np.array_equal(ga, gb)


def test_every_row_assigned():
    fold_id, _ = scaffold_folds(SMILES, n_folds=3, seed=0)
    assert set(np.unique(fold_id)).issubset({0, 1, 2})
    assert (fold_id >= 0).all()


def test_no_scaffold_spans_two_folds():
    fold_id, groups = scaffold_folds(SMILES, n_folds=3, seed=0)
    for g in np.unique(groups):
        folds_for_group = np.unique(fold_id[groups == g])
        assert len(folds_for_group) == 1, f"scaffold {g} split across folds {folds_for_group}"
