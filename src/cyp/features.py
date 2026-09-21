"""Molecular featurization: ECFP4 counts + RDKit 2D descriptors.

Deterministic given the input SMILES, so a clean checkout reproduces the same
feature matrix (and therefore the same baseline numbers).
"""

from __future__ import annotations

import numpy as np


def _descriptor_functions():
    from rdkit.Chem import Descriptors

    names = [name for name, _ in Descriptors._descList]
    funcs = [fn for _, fn in Descriptors._descList]
    return names, funcs


def feature_names(n_bits: int = 2048) -> list[str]:
    names, _ = _descriptor_functions()
    return [f"ecfp_{i}" for i in range(n_bits)] + names


def featurize(smiles, n_bits: int = 2048, radius: int = 2):
    """Return (X, valid_mask, names).

    X has shape (len(smiles), n_bits + n_descriptors). Rows for SMILES that fail
    to parse are zero and flagged False in `valid_mask`. Non-finite descriptor
    values are replaced with 0.0.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    desc_names, desc_funcs = _descriptor_functions()
    width = n_bits + len(desc_funcs)

    rows = np.zeros((len(smiles), width), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        fp = np.asarray(gen.GetCountFingerprintAsNumPy(mol), dtype=np.float32)
        desc = np.empty(len(desc_funcs), dtype=np.float32)
        for j, fn in enumerate(desc_funcs):
            try:
                v = fn(mol)
            except Exception:
                v = 0.0
            desc[j] = v if np.isfinite(v) else 0.0
        rows[i, :n_bits] = fp
        rows[i, n_bits:] = desc
        valid[i] = True

    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    return rows, valid, [f"ecfp_{i}" for i in range(n_bits)] + desc_names
