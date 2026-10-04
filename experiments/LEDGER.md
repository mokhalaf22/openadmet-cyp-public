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
| 2026-10-03T18:34:11+00:00 | phase2 | `baseline` | ρ=0.6059 (+0.0022)  ST-RAE=0.4131 (-0.0015) | 26.0m | `21d1cb5` | ok |
| 2026-10-03T19:01:33+00:00 | phase2 | `warmstart` | ρ=0.6017 (-0.0020)  ST-RAE=0.4171 (+0.0025) | 27.4m | `21d1cb5` | ok |
| 2026-10-03T19:24:11+00:00 | phase2 | `octant` | ρ=0.5940 (-0.0097)  ST-RAE=0.4214 (+0.0068) | 22.6m | `21d1cb5` | ok |
| 2026-10-03T20:01:50+00:00 | phase2 | `tox21` | ρ=0.6093 (+0.0056)  ST-RAE=0.4133 (-0.0013) | 37.7m | `21d1cb5` | ok |
| 2026-10-03T20:32:23+00:00 | phase2 | `combined` | ρ=0.6059 (+0.0022)  ST-RAE=0.4148 (+0.0002) | 30.6m | `21d1cb5` | ok |
| 2026-10-03T21:20:55+00:00 | phase2b | `ws_lowlr` | ρ=0.5875 (-0.0162)  ST-RAE=0.4235 (+0.0089) | 20.1m | `1f312ed` | ok |
| 2026-10-03T21:41:08+00:00 | phase2b | `ws_freeze` | ρ=0.5885 (-0.0152)  ST-RAE=0.4233 (+0.0087) | 20.2m | `1f312ed` | ok |
| 2026-10-04T13:18:15+00:00 | phase3 | `proxy_targets` | — | 0.5m | `9501050` | ok |
| 2026-10-04T13:34:11+00:00 | phase3 | `proxy` | ρ=0.5918 (-0.0119)  ST-RAE=0.4573 (+0.0427) | 15.9m | `9501050` | ok |
| 2026-10-04T14:02:11+00:00 | phase3 | `ci_mc` | ρ=0.6009 (-0.0028)  ST-RAE=0.4196 (+0.0050) | 28.0m | `9501050` | ok |
| 2026-10-04T15:05:12+00:00 | arch | `allheads` | — | 0.1m | `dd670ff` | failed |
| 2026-10-04T15:26:47+00:00 | arch | `allheads` | ρ=0.5995 (-0.0042)  ST-RAE=0.4200 (+0.0054) | 19.7m | `ecc75b8` | ok |
| 2026-10-04T15:49:55+00:00 | arch | `per_isoform` | ρ=0.5870 (-0.0167)  ST-RAE=0.4249 (+0.0103) | 23.1m | `ecc75b8` | ok |

## Per-isoform detail

**phase2 / baseline**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.530 | 0.507 | 0.034 |
| CYP2C9 | 0.668 | 0.307 | 0.017 |
| CYP2D6 | 0.442 | 0.568 | 0.033 |
| CYP3A4 | 0.783 | 0.270 | 0.017 |

**phase2 / warmstart**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.525 | 0.511 | 0.038 |
| CYP2C9 | 0.668 | 0.311 | 0.020 |
| CYP2D6 | 0.430 | 0.574 | 0.035 |
| CYP3A4 | 0.783 | 0.271 | 0.019 |

**phase2 / octant**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.514 | 0.519 | 0.036 |
| CYP2C9 | 0.662 | 0.310 | 0.022 |
| CYP2D6 | 0.420 | 0.581 | 0.026 |
| CYP3A4 | 0.780 | 0.275 | 0.012 |

**phase2 / tox21**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.536 | 0.505 | 0.035 |
| CYP2C9 | 0.668 | 0.309 | 0.018 |
| CYP2D6 | 0.450 | 0.570 | 0.038 |
| CYP3A4 | 0.783 | 0.269 | 0.020 |

**phase2 / combined**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.529 | 0.513 | 0.031 |
| CYP2C9 | 0.672 | 0.306 | 0.017 |
| CYP2D6 | 0.439 | 0.573 | 0.036 |
| CYP3A4 | 0.784 | 0.268 | 0.020 |

**phase2b / ws_lowlr**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.519 | 0.515 | 0.035 |
| CYP2C9 | 0.651 | 0.316 | 0.024 |
| CYP2D6 | 0.408 | 0.586 | 0.029 |
| CYP3A4 | 0.773 | 0.277 | 0.014 |

**phase2b / ws_freeze**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.520 | 0.514 | 0.034 |
| CYP2C9 | 0.650 | 0.317 | 0.024 |
| CYP2D6 | 0.411 | 0.585 | 0.026 |
| CYP3A4 | 0.772 | 0.278 | 0.014 |

**phase3 / proxy**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.518 | 0.595 | 0.095 |
| CYP2C9 | 0.659 | 0.335 | 0.027 |
| CYP2D6 | 0.414 | 0.623 | 0.026 |
| CYP3A4 | 0.776 | 0.276 | 0.013 |

**phase3 / ci_mc**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.526 | 0.509 | 0.035 |
| CYP2C9 | 0.666 | 0.314 | 0.021 |
| CYP2D6 | 0.430 | 0.575 | 0.034 |
| CYP3A4 | 0.781 | 0.280 | 0.014 |

**arch / allheads**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.524 | 0.514 | 0.033 |
| CYP2C9 | 0.663 | 0.315 | 0.027 |
| CYP2D6 | 0.432 | 0.577 | 0.029 |
| CYP3A4 | 0.779 | 0.275 | 0.020 |

**arch / per_isoform**  
| isoform | Spearman | ST-RAE | ST-RAE fold std |
|---|---|---|---|
| CYP1A2 | 0.516 | 0.519 | 0.038 |
| CYP2C9 | 0.649 | 0.320 | 0.033 |
| CYP2D6 | 0.404 | 0.585 | 0.027 |
| CYP3A4 | 0.779 | 0.275 | 0.014 |

