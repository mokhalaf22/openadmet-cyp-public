"""Scaffold cross-validation splits.

The blind/leaderboard split is made by chemical series (75 potent parents plus
their nearest analogs), so random CV puts near-duplicates on both sides and
reports a number that will not survive submission. We group by Bemis-Murcko
scaffold and keep every analog of a scaffold on one side of every fold.

The assignment is deterministic given the input SMILES order (GroupKFold does
not shuffle), so a clean checkout reproduces identical folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES; empty string on parse failure."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def scaffold_groups(smiles) -> np.ndarray:
    """Integer group id per molecule, one per distinct Murcko scaffold."""
    scaffolds = [murcko_scaffold(s) for s in smiles]
    return pd.factorize(pd.Series(scaffolds))[0]


def scaffold_folds(smiles, n_folds: int = 5, seed: int = 0):
    """Return (fold_id, groups).

    `fold_id[i]` is the validation fold (0..n_folds-1) that molecule i belongs
    to; `groups[i]` is its scaffold id. Whole scaffolds are assigned to a single
    fold, so no scaffold appears in both train and validation.

    GroupKFold is deterministic and ignores `seed`; the parameter is kept for a
    stable interface and future seed-dependent splitters.
    """
    from sklearn.model_selection import GroupKFold

    groups = scaffold_groups(smiles)
    n = len(groups)
    fold_id = np.full(n, -1, dtype=int)
    gkf = GroupKFold(n_splits=n_folds)
    dummy_X = np.zeros((n, 1))
    for fold, (_, val_idx) in enumerate(gkf.split(dummy_X, groups=groups)):
        fold_id[val_idx] = fold
    return fold_id, groups
