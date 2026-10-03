"""Phase definitions for experiments/runner.py.

Each phase is an ordered list of steps plus the gate that ends it. The runner never advances
past a gate. A step is:

    {
      "name":        short id, used as the ledger key and log name
      "run":         shell command (optional; omitted for pure-metric steps)
      "done_when":   artifact path whose existence means the step is complete (checkpoint)
      "metrics":     "module:callable" returning a dict recorded in the ledger (optional)
      "metrics_args" kwargs for that callable (optional)
      "config":      free-form dict recorded in the ledger for provenance
    }

Phases 2-4 are declared so the plan is visible and reviewable, but each is fenced behind
`blocked` until its predecessor gate has been reviewed and its steps wired — the runner refuses
to run a fenced phase rather than half-executing it.
"""

PY = "./.venv/bin/python"


def _blocked(reason: str) -> list[dict]:
    return [{
        "name": "blocked",
        "run": f"bash -c 'echo {reason!r}; exit 1'",
        "config": {"blocked_reason": reason},
    }]


PHASES: dict[str, dict] = {
    # ------------------------------------------------------------------ 1 ---
    "phase1": {
        "title": "retrieval of unlabelled near neighbours of the 750 blinded compounds",
        "steps": [
            {
                "name": "deps",
                "run": f"{PY} -m pip install -q smallworld-api useful_rdkit_utils",
                "config": {
                    "packages": ["smallworld-api", "useful_rdkit_utils"],
                    "note": ("rd_filters is not on PyPI (GitHub-only); useful_rdkit_utils ships "
                             "the same REOS alert catalogs (Glaxo, Dundee, BMS, PAINS, SureChEMBL, "
                             "MLSMR, Inpharmatica, LINT) with per-rule drop_rule(), which the "
                             "subtractive veto needs."),
                },
            },
            {
                "name": "sw_probe",
                "run": f"{PY} experiments/sw_retrieve.py --probe",
                "done_when": "experiments/sw_cache/_probe.json",
                "metrics": "sw_retrieve:probe_metrics",
                "config": {"route": "SmallWorld API (fallback route)",
                           "why": "primary 15.26B-SMILES local search needs HPC + TB storage"},
            },
            {
                "name": "sw_retrieve",
                "run": f"{PY} experiments/sw_retrieve.py --all",
                "done_when": "experiments/sw_cache/_complete",
                "metrics": "sw_retrieve:retrieval_metrics",
                "config": {"queries": 750, "batched": True, "resumable": "per-compound cache"},
            },
            {
                "name": "corpus_filter",
                "run": f"{PY} experiments/corpus_filter.py",
                "done_when": "experiments/corpus_phase1.csv",
                "metrics": "corpus_filter:gate1_metrics",
                "config": {
                    "envelope": "physicochemical envelope of the blinded set",
                    "alerts": "rd_filters with subtractive veto (drop any alert firing on a blinded compound)",
                    "exclusions": "6 quarantined Octant overlaps + challenge train/test",
                    "target_corpus": "30k-90k admitted",
                },
            },
        ],
        "gate": (
            "GATE 1 — report admitted corpus size, blind anchor density @ Tanimoto 0.7, and median\n"
            "nearest-neighbour Tanimoto (blinded -> corpus). Prior ChEMBL/PubChem attempt: 324\n"
            "compounds / 13.3% density (too thin). STOP if under ~10,000 compounds or under 40%\n"
            "anchor density — the rest of the plan is premised on retrieval working."
        ),
        "gate_report": "corpus_filter:gate1_report",
    },

    # ------------------------------------------------------------------ 2 ---
    "phase2": {
        "title": "encoder warm start + external heads (Octant CYP3A4, Tox21), 4 attribution runs",
        "steps": [
            {   # reference leg: must reproduce 0.6037 / 0.4146, which validates the trainer
                "name": "baseline",
                "run": f"{PY} experiments/phase2_train.py --leg baseline",
                "done_when": "experiments/p2_baseline_oof.npy",
                "metrics": "runner:oof_metrics",
                "metrics_args": {"npy_path": "experiments/p2_baseline_oof.npy"},
                "config": {"leg": "baseline", "expect": "0.6037 / 0.4146"},
            },
            {
                "name": "warmstart",
                "run": f"{PY} experiments/phase2_train.py --leg warmstart",
                "done_when": "experiments/p2_warmstart_oof.npy",
                "metrics": "runner:oof_metrics",
                "metrics_args": {"npy_path": "experiments/p2_warmstart_oof.npy"},
                "config": {"leg": "warmstart", "corpus": 111361,
                           "pretrain": "physchem-only (label-free, shared across folds)"},
            },
            {
                "name": "octant",
                "run": f"{PY} experiments/phase2_train.py --leg octant",
                "done_when": "experiments/p2_octant_oof.npy",
                "metrics": "runner:oof_metrics",
                "metrics_args": {"npy_path": "experiments/p2_octant_oof.npy"},
                "config": {"leg": "octant", "aux_rows": 1083,
                           "note": "CYP3A4 combined reversible+TDI condition, separate head"},
            },
            {
                "name": "tox21",
                "run": f"{PY} experiments/phase2_train.py --leg tox21",
                "done_when": "experiments/p2_tox21_oof.npy",
                "metrics": "runner:oof_metrics",
                "metrics_args": {"npy_path": "experiments/p2_tox21_oof.npy"},
                "config": {"leg": "tox21", "aux_rows": 7879,
                           "isoforms": "CYP1A2/2C9/2D6 (no CYP3A4 in panel)"},
            },
            {
                "name": "combined",
                "run": f"{PY} experiments/phase2_train.py --leg combined",
                "done_when": "experiments/p2_combined_oof.npy",
                "metrics": "runner:oof_metrics",
                "metrics_args": {"npy_path": "experiments/p2_combined_oof.npy"},
                "config": {"leg": "combined", "legs": ["warmstart", "octant", "tox21"]},
            },
        ],
        "gate": (
            "GATE 2 — OOF Spearman and ST-RAE per isoform for: warm-start alone, Octant alone,\n"
            "Tox21 alone, and combined (four runs, so each source is attributable), against\n"
            "D-MPNN+primary (0.6037 / 0.4146). Seed floor ~0.004."
        ),
    },

    # ------------------------------------------------------------------ 3 ---
    "phase3": {
        "title": "multi-fidelity + metric-aligned training: proxy rows, CI sampling, SMILES enumeration",
        "steps": _blocked("phase3 is fenced until GATE 2 is reviewed and its steps are wired"),
        "gate": (
            "GATE 3 — report (a) two-stage proxy-label rows, (b) credible-interval MC sampling,\n"
            "(c) ~20x SMILES enumeration with test-time averaging, each against baseline.\n"
            "Recommend at most one for a board slot."
        ),
    },

    # ------------------------------------------------------------------ 4 ---
    "phase4": {
        "title": "multi-task imputation + non-negative stacking over surviving legs",
        "steps": _blocked("phase4 is fenced until GATE 3 is reviewed and its steps are wired"),
        "gate": (
            "GATE 4 — final candidate vs the current submission: better or not, and on what\n"
            "evidence. Stacking selected on cluster-disjoint folds, PCA-capped tabular inputs."
        ),
    },
}
