# Experiment ledger

Baseline for model steps: macro OOF Spearman **0.6037**, ST-RAE **0.4146** (D-MPNN + predicted-primary, GFOLD). Seed floor ≈ 0.004 — smaller deltas are noise.

| when (UTC) | phase | step | result | wall | commit | status |
|---|---|---|---|---|---|---|
| 2026-10-03T03:16:13+00:00 | phase1 | `deps` | — | 0.0m | `96f545e` | failed |
| 2026-10-03T03:18:28+00:00 | phase1 | `deps` | — | 0.0m | `96f545e` | ok |
| 2026-10-03T03:18:48+00:00 | phase1 | `sw_probe` | probe_hits=200  probe_ecfp4_max=0.6818181818181818  db=REALDB-2025-07.smi.anon  dist=4  length=200 | 0.3m | `96f545e` | ok |
| 2026-10-03T14:49:49+00:00 | phase1 | `sw_retrieve` | queries_cached=750  queries_empty=8  raw_hits=146687  unique_hit_smiles=131989 | 160.5m | `d828471` | ok |
| 2026-10-03T15:20:07+00:00 | phase1 | `corpus_filter` | corpus_size=57191  anchor_density_0.7=0.644  median_nn_tanimoto=0.766  frac_nn_ge_0.5=0.888  passes_gate=True | 30.3m | `d828471` | ok |
| 2026-10-03T15:31:30+00:00 | phase1 | `corpus_filter` | corpus_size=111361  anchor_density_0.7=0.9027  median_nn_tanimoto=0.8182  frac_nn_ge_0.5=0.9747  passes_gate=True | 9.0m | `f911b64` | ok |
