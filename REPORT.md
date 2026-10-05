# OpenADMET CYP Inhibition Blind Challenge — Model Report

This is the public model report for our entry. Numbers are **out-of-fold (OOF) on scaffold
splits** unless labelled **blind** (the live leaderboard, scored on half the held-out set).
Full derivations, tables, and plots are in [`FINDINGS.md`](FINDINGS.md); this document is the
argument and the rationale.

## Thesis

**The ceiling on this benchmark is representational, not a matter of model capacity or tuning.**
A learned graph representation plus **one** well-chosen derived feature moved the floor as far as
public data allows; separately, **post-hoc calibration** delivered the actual leaderboard gains.
The remaining gap to the leaders is a property of the **data**, not the method: the blinded
compounds are Enamine make-on-demand chemistry with almost no public measured data, so every
external-data and ranking-transfer lever fails for the same reason.

- **Regression (blind, live board):** MA-ST-RAE **0.9356 → 0.7114 → 0.6683 → 0.6411** (rank
  **200 → 95 → 83 → 77**) as we added the final model, a validated CYP2D6 location correction, and
  two dispersion calibrations. Current board: MAE 0.8394, R² 0.3477, Spearman 0.6965 — ranking is
  at the representational ceiling; the gains are calibration.
- **TDI classification (blind):** MA-MCC **0.273 → 0.3097** (rank **73 → 59**) after a threshold
  fix. Tightening further *lost* MCC — the leaders win by discrimination, not by a tighter cut.
- **What moved the model:** the interval formulation (recovered a CYP3A4 gap capacity could not)
  and a structure-**predicted primary-screen** feature (the one auxiliary lever that worked).
- **What did not, and why:** bigger models, foundation-model fine-tuning, auxiliary assays,
  difference learning, and external neighbours were all null — because the relevant public data
  does not exist (§5).
- **Test-set leakage in a same-lab public release (a finding in its own right).** A pre-ingest
  structure-level check found **5 of the 750 blinded test compounds present in the Octant release**
  (same lab as the challenge) by exact structure and OCNT identifier, with CYP3A4 labels attached;
  the release is 99.3% the challenge's own compounds. All 6 overlaps (incl. one near-duplicate)
  were quarantined and the overlap disclosed to the organizers; nothing was built from it (§9).
- **No proprietary data.** Public sources (AID 1851, ChEMBL/PubChem, the Octant release) were
  examined and disclosed.

## 1. Data, provenance, and conventions

- **Source:** `openadmet/cyp-challenge-train-test`, pinned at revision
  `3ac9c5dbb83eec5780ec7fa511908698cfe1396d` (2026-08-27), fetched at that exact SHA by
  `src/cyp/download.py` — reproducible from a clean checkout. Training uses
  `cyp-challenge-TRAIN_TDI.csv` (6,145 rows, both arms) as the single source; the blinded file
  provides SMILES + `Molecule_Name` only.
- **Conventions verified before modelling** (FINDINGS §1–§5): censoring is already inside the
  reported credible intervals (consumed verbatim as `[lo, hi]`; no `(-inf, 4.0]` special case);
  `TRAIN_TDI` is byte-identical to and a superset of `TRAIN_inhibition`; a *missing* direct arm
  means "not assayed," not "censored," and those rows are excluded from TDI train/eval via
  `src/cyp/guards.py`.
- **Featurization:** ECFP4 counts + RDKit 2D descriptors, with degenerate columns (`Ipc`,
  near-zero-variance, non-finite) dropped before standardizing (FINDINGS §11); a chemprop D-MPNN
  graph encoder for the learned representation.

## 2. The ceiling is representational (the linearity floor)

On this feature set the signal is **essentially linear**: ridge regression ≈ LightGBM (macro OOF
ST-RAE 0.489 vs 0.451), and a well-optimized MLP does no better (FINDINGS §11). Extra model
capacity does not help. So the lever is never "a bigger model" — it is the **representation** and
the **loss**. This single fact frames the whole entry: the two things that *did* move the floor
are a better loss (§3) and a better representation + one feature (§4); everything that tried to
add capacity or external breadth (§5) did not.

## 3. The interval formulation — recovering what capacity could not

Every target is an interval, not a point. The loss is an **interval hinge**: L1 distance to the
nearest credible-interval bound, zero inside the interval (`src/cyp/losses.py`) — matching the
competition's soft-threshold metric exactly, so training and scoring agree.

Switching point → interval targets improved macro OOF ST-RAE 0.445 → 0.434 and **recovered the
CYP3A4 gap** (0.323 → 0.305, to LightGBM parity; FINDINGS §14). CYP3A4 is precisely the isoform
with **23.6% of rows wider than 2 log units**, and its gap had survived *every* capacity lever —
more epochs, more width, GBM vs MLP, from-scratch vs pretrained (§13, §32). Scoring against the
reported bounds, rather than a point the assay never measured, is what closed it. This is the
entry's core statistical argument: a formulation matched to the data's uncertainty, not a larger
network, moved the floor.

**Validated a second, independent way: by what dilutes it.** The formulation was first confirmed by
what *improved* the model (point → interval targets, above). It was confirmed again by two attempts
to add information that failed *because* they weakened the interval treatment (FINDINGS §44):

- **Proxy supervision.** The single-concentration screen let us fill sparse cells of the isoform
  matrix: **11,505 proxy-supervised (compound, isoform) cells against 6,525 real ones**, from
  per-fold `log2FC + structure → pIC50` mappings of genuinely good quality (held-out Pearson
  **0.901 / 0.860 / 0.756 / 0.933**). It still cost **ST-RAE +0.0442** against Spearman **−0.0141**.
- **Within-interval sampling.** Drawing the training target from inside each reported `[lo, hi]`
  (one draw per epoch) instead of using the hinge: **ST-RAE +0.0065**. Same failure, milder.

**The asymmetry is the finding, not the sign.** Ranking barely moved while ST-RAE degraded ~3×
more, because proxy cells are necessarily *point* targets: 11,505 of them — nearly 2× the real
supervision even at weight 0.3 — pull the model back toward point regression and destroy the
hinge's **zero-error zone inside the reported interval**, which *is* the metric. The information
was accurate; its *fidelity* was wrong.

> **General rule: more supervision at the wrong fidelity is worse than less supervision at the
> right one.** When the metric scores against intervals, a confident point target is not a weaker
> version of an interval — it is a different and conflicting claim about what was measured.

**Validated a third way, and this one reframes the hinge from default to derivation.** The hinge
weights every row equally. Because 85% of our ST-RAE came from the three narrowest interval
quartiles (FINDINGS §46b), reweighting the loss toward them looked obviously right. We swept the
weight exponent p in `1/(1+width)^p` over five values, **both** directions (FINDINGS §49, §49a):

| p | −1 | −0.5 | **0 (uniform)** | +1 | +2 |
|---|---|---|---|---|---|
| macro OOF ST-RAE | 0.4311 | 0.4177 | **0.4131** | 0.4197 | 0.4391 |

A V with its minimum exactly at uniform. The dial itself worked perfectly — monotone in p within
every quartile, and Q4's inside-interval rate moved 20.7% → 47.3% across the range — so this is
not a mechanism that failed to engage. It is a reallocation that has nothing to gain.

The reason is structural. ST-RAE's numerator is a **plain sum** of per-row excursions over a
prediction-independent denominator, so `∂ST-RAE/∂pᵢ = ±1/den` for every excursing row **regardless
of its interval width**: the metric weights rows *equally*. "85% of the error sits in the narrow
quartiles" describes where error **lands** — narrow intervals are simply harder to land inside —
not how the metric **weights** rows. Those are different quantities, and we had conflated them.

> **General rule: when a metric's per-row gradient is uniform, match it. Where the error
> concentrates is not a weighting signal.** The uniform interval hinge is therefore not a default
> we got away with — it is the loss whose per-row gradient matches the scored metric exactly, and
> any reweighting in either direction deliberately mismatches that gradient and must lose in
> expectation. Five runs bound the whole family; there is no p worth tuning.

## 4. The one auxiliary lever that worked — the predicted primary screen

The single-concentration screen (log2FC) is a **different assay** from the scored dose-response
pIC50. Its raw values are test-unavailable, but a model that **predicts** log2FC from structure is
test-available. Trained out-of-fold and added as four features, it is the only auxiliary signal
that improved the underlying model:

- On LightGBM: macro OOF ST-RAE 0.451 → 0.436. **On the D-MPNN: 0.434 → 0.415** (macro OOF
  Spearman 0.60), the largest single model gain in the entry (FINDINGS §25, §26a).

It works for two reasons that turn out to be the general rule here: the surrogate is **strongly
learnable from structure** (OOF Pearson 0.59–0.74) and it carries **new, different-assay
information**.

**A factual correction worth stating, because it explains the mechanism** (FINDINGS §44): there are
**no screen-only molecules.** All **4,376** unique single-concentration compounds are already in the
dose-response table (matched by InChIKey; exactly one differs by SMILES string alone). The screen is
therefore **a second readout on the same compounds, not an additional pool of compounds** — which is
precisely why it pays as a *feature* (extra information per compound) and not as extra training
*rows* (there are none to add). An earlier plan to add "the ~4,376 screen-only molecules" as rows had
no molecules to act on; retargeting it to fill sparse matrix *cells* instead then failed for the
fidelity reason in §3. The **final regression model is this D-MPNN + predicted-primary** (multi-task over
the four isoforms, interval targets, 3-seed ensemble). A CheMeleon foundation encoder — frozen
(embeddings → TabICL) or fine-tuned end-to-end — did **not** beat it (FINDINGS §26, §32),
consistent with §2: the from-scratch D-MPNN is already at the representational ceiling for this
data.

## 5. Why every other lever failed at once — chemical isolation

The blinded set is **Enamine make-on-demand catalogue chemistry that public bioactivity databases
barely cover.** Three independent measurements (FINDINGS §13, §35):

- nearest-neighbour ECFP4 Tanimoto to **PubChem AID 1851** (17k-compound CYP panel): **median
  0.368**;
- **ChEMBL + PubChem** 70%-similarity search across all 750 blinded compounds: only **324 unique
  neighbours** (many queries return zero);
- **blind anchor density at Tanimoto 0.7: 13.3%** — 87% of the blind set has no near neighbour
  even after retrieval;
- even the **same-lab Octant release** is **99.3% the challenge's own compounds** (§9), not new
  data.

This one fact explains a otherwise-confusing list of nulls as a **single cause**: the AID 1851
auxiliary heads *degraded* the model (distant chemistry pulls the shared encoder off ours,
FINDINGS §19); a predicted-**Emax** surrogate was null (same assay, only weakly learnable, §31a);
cross-model blending was below noise once the primary feature made the models converge (§31b); a
pretrained CheMeleon fine-tune was null (§32); an explicit **pairwise ranking loss** was null
(§33a); **SQRL/DeepDelta difference learning** could not be built at all — it needs dense near
neighbours, and there are none (13.3% anchor density; §33b); and a physicochemical
near-neighbour **warm-start** was null because the retrievable corpus is only 324 compounds
(§35). **The methods were not wrong — the compounds that would inform the blind set have no public
measured data.** Within the public-data envelope, the model's ordering is at its representational
ceiling (OOF Spearman ≈ 0.604; blind 0.6965; leaders ≈ 0.78), and closing the rest would require
measured data on this specific Enamine expansion, which is not publicly available.

### 5a. A stronger negative: the data existed, was close, and still did not help

§5 says the useful public data does not exist. We then tested the harder version of that claim by
**manufacturing** the missing data, and it still failed — which is the more informative result.

Using the SmallWorld API against Enamine REAL we retrieved **111,361 unlabelled compounds** around
the 750 blinded structures (after excluding challenge train/test, 6 quarantined overlaps, a
physicochemical-envelope filter, and an alert screen with a subtractive veto so no alert firing on a
blinded compound could prune the corpus). The corpus is genuinely dense where it matters:

| | median NN Tanimoto | ≥0.7 |
|---|---|---|
| blinded → corpus | **0.818** | **90.3%** |
| training → corpus | 0.406 | 2.2% |
| *(for scale)* training → blinded | 0.294 | 0.7% |

We pretrained the D-MPNN encoder on it with a computed-physicochemical head (no measured label ever
touches a retrieved compound), then fine-tuned on the DRC targets. Results, against an in-run
control, 3 seeds, scaffold folds (FINDINGS §40–§43):

- **warm start: −0.0042 macro Spearman** (no gain);
- a per-epoch trajectory probe showed the warm start *starts ahead* (epoch-1 inner-validation
  −0.0075) but is **erased by epoch 2**;
- so we held the encoder in place two ways — encoder LR at 1/10, and frozen 5 epochs then released —
  and **both were markedly worse: −0.0184 and −0.0174 macro Spearman**, worse on every isoform.

**Holding the pretrained representation cost more than letting the fine-tune overwrite it.** The
fine-tune was therefore not destroying something valuable; it was correctly discarding a
representation that does not serve the task. The early advantage was *conditioning*, not transfer.

**What this establishes, stated plainly so nobody repeats it:**
1. **Unlabelled structural density is not the information this task lacks.** 111k compounds at
   median 0.818 to the test set changed nothing. The missing quantity is *measured potency*, and
   structural proximity is not a substitute for it.
2. **Nearby chemistry does not substitute for measured labels.** Retrieval difficulty was never the
   bottleneck — we solved retrieval (90.3% anchor density, up from 13.3% via ChEMBL/PubChem) and the
   bottleneck did not move.
3. **An "erased warm start" is not automatically a warm start worth protecting.** Before spending
   effort on frozen layers or discriminative learning rates, test whether holding the representation
   helps — here it hurt by 4x the amount the erasure did.

This is a stronger negative than §5 because the usual escape hatch ("you simply lacked the data") is
closed: the data existed, it was close, it was clean, and it still did not help.

## 6. Calibration — a separate, validated contribution

Post-hoc calibration cannot change *ranking*, but it drove the actual leaderboard gains, and its
story is a reusable lesson in OOF-vs-blind divergence.

- **OOF underestimated blind error ~2×.** Scaffold-split OOF macro ST-RAE was 0.451 while the
  first blind reveal was **0.9356**: ranking was preserved (Spearman 0.6336) but magnitude
  collapsed (R² ≈ 0). Cause: L1/interval regression predicts the compressed conditional median,
  and the blind set sits in a different potency regime — so predictions shrink to near the
  predict-a-constant baseline (FINDINGS §9, §22).
- **CYP2D6 location shift (−0.5): validated.** The organizers confirmed the CYP2D6 test compounds
  are less potent than training (the test excluded CYP2D6 hit-expansion). A −0.5 location
  correction — validated on a shifted-eval simulation (§24) — directly targets this and was
  confirmed on the board.
- **Dispersion: an OOF-unjudgeable gamble that paid, twice.** Because dispersion preserves ranking,
  OOF (which we score on ranking-sensitive ST-RAE at fixed scale) structurally cannot evaluate it.
  Expanding predictions toward the training spread, ranking held fixed (Spearman invariant), cut
  blind MA-ST-RAE **0.7114 → 0.6683** (0.85× spread; rank 95 → 83), and a second expansion to full
  training spread on the three isoforms with no known distribution shift (CYP2D6 held gentler, per
  §24) cut it further to **0.6411** (rank 83 → 77; MAE 0.8394, R² 0.2495 → **0.3477**, Spearman
  unchanged at 0.6965; FINDINGS §30, §30a). Compression was costing us, not encoding honest
  uncertainty.

**The two metrics want different spreads, and the gap is real but not exploitable.** Decomposing
`R² = 2ρk − k² − b²` (k = sd(pred)/sd(true), b = mean offset in sd(true) units) makes the
R²-optimal spread **k = ρ, not k = 1**. Inverting our own board score gave k = 0.7108 against
ρ = 0.6965 — on the optimum — and attributed **100% of the remaining R² headroom to location**
(b = +0.48 log units; FINDINGS §46). Acting on it moved blind **R² 0.3477 → 0.4272** and
**ST-RAE 0.6411 → 0.6404**: the algebra was right and the scored metric did not care.

That divergence is structural, and we measured it from the spread axis too (FINDINGS §52). Swept
per isoform on OOF with location re-fit at every k, the **ST-RAE-optimal spread sits *below* the
R²-optimal k = ρ — about 0.8ρ in-distribution** (CYP1A2 0.42, CYP2C9 0.56, CYP2D6 0.33, CYP3A4
0.63 against ρ of 0.53/0.67/0.44/0.78). So a model calibrated for R² is mis-calibrated for ST-RAE
by construction. **But the gap is not exploitable:** the ST-RAE-vs-k curve is *flat at its
minimum* — our raw model sits at k = 0.497 against an optimum of 0.48, a difference of **0.0002
ST-RAE** — and k\*/ρ is not a transportable constant (0.50–0.65 within strata vs 0.80 on the full
population), because k\* is a property of the evaluation population's composition, not of the
metric alone.

> **Two populations, not one contradiction.** OOF's in-distribution sd(true) is 0.78–1.09 per
> isoform; the blind population's, backed out of our own score, is **1.297** — 36% wider than our
> label sd of 0.956, because a label set built only from successful curve fits silently excludes
> non-inhibitors. The same k\* *fraction* therefore implies a much larger *absolute* spread on the
> blind set. This is why OOF said expanding spread **hurt** every isoform (§24) while the board
> said it **gained 0.043** (§30) — and **the board was right.** We did not act on an OOF-derived
> compression argument a second time (§52): the same reference error, caught twice, is a
> methodology failure rather than bad luck.

We codified a **live-board discipline** to avoid overfitting the scored half (FINDINGS §31c): OOF
is a weak prior (calibrated on only two independent points), act only on board moves above the
split standard error (~0.03), require OOF-and-board agreement for the final pick, and treat the
unscored half as a permanent holdout. Only ~0.04-scale moves are trustworthy on a half-set board.

## 7. TDI classification, and the derived-label refutation

The architecture predicts `mu` (direct pIC50) and `delta ≥ 0` (the preincubation shift), so the
TDI-arm pIC50 is `mu + delta` by construction. On the regression side this two-head sharing is
free (it does not hurt direct ST-RAE). But the headline idea — deriving the boolean TDI label
*for free* from the two arms — **does not hold up** (FINDINGS §15): a smooth regression `delta`
cannot reproduce the sharp 0.301 threshold crossings CYP2D6's boundary-dominated positives
require, and the derived label collapses (CYP2D6 MCC ≈ 0.03). **A plain classifier head wins**
(CYP2D6 0.125 vs 0.031), and it must share the regression encoder — a standalone TDI-only encoder
is worse (0.111), so the regression data regularizes it (§20). This is an honest negative result
on the report's own proposed innovation.

On the board, a threshold fix (CYP2D6 0.30, CYP3A4 0.45) lifted MA-MCC **0.273 → 0.3097** (rank
73 → 59; FINDINGS §29). Tightening CYP3A4 further (to 0.65) then *lost* MCC — precision rose but
recall fell more (§30). The lesson: threshold moves cannot substitute for a better-ranked
classifier; the leaders reach their operating point by discrimination, not a tighter cut.

**The shared encoder is a measured net transfer from regression to classification** (FINDINGS §45).
Both directions have now been tested, and they are not symmetric:

- **Classification gains from sharing.** A standalone TDI-only encoder is *worse* than the shared
  one (CYP2D6 MCC 0.111 vs 0.125; §20) — the regression data regularises the encoder.
- **Regression pays for it.** Adding the supervised delta head (TDI-arm intervals as `mu + delta`)
  and the TDI classifier head to the regression model costs **−0.0064 macro Spearman**, beyond the
  seed floor and **consistent across all four isoforms** (−0.004 to −0.010).

**This is why the two tracks ship from separate models**, which costs nothing because the
competition takes two independent submission files: regression comes from a regression-only model
(four interval heads, nothing else supervised), classification from the shared one. The point worth
stating is that this configuration is **tested, not assumed** — we measured the cost in both
directions rather than inheriting the architecture from the original plan.

## 8. What to reuse, and what not to bother trying

**Reuse on this benchmark:**
- the **interval-hinge loss** matched to the credible intervals (recovers wide-interval isoforms);
- a **learned D-MPNN representation** over ECFP/descriptors (the representation is the lever, §2);
- the **predicted-primary-screen** feature — and the general rule it embodies: a surrogate helps
  only when it is *strongly learnable from structure* **and** from a *different assay*;
- **post-hoc dispersion** + a **validated location shift** for a known target-distribution shift,
  tested on the board under the §6 discipline;
- **fidelity over volume whenever the metric scores against intervals** (§3): match the *form* of
  the supervision to the metric before chasing more of it. Here 11,505 accurate-but-point proxy
  labels were worse than 6,525 interval ones;
- **shared heads across the isoforms, but separate models per track** (§7, FINDINGS §45). Splitting
  the four isoforms into single-task models costs −0.0189 macro Spearman, so sharing earns its
  place. The *mechanism* is the part that generalises: the benefit of sharing scales **inversely
  with per-task data volume** —

  | isoform | ρ cost of splitting | rows |
  |---|---|---|
  | CYP2D6 | **−0.038** | 1,493 |
  | CYP2C9 | −0.019 | 1,285 |
  | CYP1A2 | −0.014 | 1,412 |
  | CYP3A4 | −0.004 *(within noise)* | 2,335 |

  Sharing subsidises the data-poor tasks from the data-rich one, so **it matters most exactly where
  the data is thinnest** — and is close to free to give up on the task that has enough. Conversely,
  share across *tasks of the same kind*, not across *tracks*: the regression/classification sharing
  is a net transfer that the paying side should opt out of (§7);
- **a structure-level leakage check before ingesting any external data** — and, as a standing
  rule: **when a challenge is run by a lab that also publishes datasets, check its public releases
  for the blinded test compounds (by canonical structure/InChIKey, not just identifiers) before
  using anything.** Here that check found 5 test compounds with labels in a same-lab release (§9);
  quarantine every match and near-duplicate, and disclose.

**Do not bother (on this benchmark, with public data):**
- distant external assays as auxiliary heads (AID 1851 — chemical-space overlap gates usefulness);
- foundation-model fine-tuning over a from-scratch D-MPNN (no gain at this label count);
- same-assay surrogate features (predicted-Emax — weakly learnable, no new information);
- cross-model blending once the primary feature has converged the models;
- explicit ranking losses (pairwise margin — the bottleneck is representational, not the loss);
- SQRL/difference learning or neighbour warm-starts **without** a dense external neighbour corpus
  — the retrievable public corpus is 324 compounds at 13.3% anchor density;
- the Octant release as "external" data — it is 99.3% the challenge's own compounds;
- **proxy point targets to fill the sparse isoform matrix** — not because the proxies are
  inaccurate (held-out Pearson 0.76–0.93) but because a point target contradicts the interval the
  assay actually reported, and the metric scores intervals: ST-RAE +0.0442 (§3);
- **sampling the target from inside the reported interval** — it discards the hinge's agreement
  with the metric and adds variance for nothing: ST-RAE +0.0065 (§3);
- **reweighting the loss toward where the error concentrates** — the metric's per-row gradient is
  already uniform, so any weighting mismatches it; five exponents bound the family and uniform wins
  (§3, FINDINGS §49a);
- **ensembling, in the regime where only one of your models is good** (FINDINGS §48). This one is
  worth stating as a precondition rather than a result, because the published gains are real and
  simply do not transfer:

  > **Averaging pays only when members are *both* mutually diverse *and* comparably strong.** Those
  > are two requirements, not one, and they trade off: **every diversity lever buys diversity by
  > spending accuracy** — a different model class, dropping a shared feature, resampling the
  > training set. We measured the trade directly on CYP2D6: the average's gain correlated **+0.892
  > with member strength** and **−0.736 with member decorrelation**, and the only member that helped
  > at all was the *most* correlated one (ρ = 0.920), because it was the strongest.

  We could reach ρ = 0.758 between members — genuine decorrelation — and the average still gained
  **+0.0002**. Our alternative model classes (LightGBM, TabICL, CheMeleon) are all materially weaker
  than the D-MPNN, so the diverse-*and*-strong member does not exist for us to average in. **The
  practical pre-test is not "do my members differ?" but "do I have two or more independently built
  models of comparable accuracy?"** If not, the effort belongs in making a second model strong, not
  in combining weak ones.

## 9. Limitations, reproducibility, and external-data disclosure

- **All model numbers are scaffold-split OOF; blind = the live half-set board.** OOF is a usable
  proxy for *ordering* but not *calibrated magnitude* here (§6); size confidence to the size of
  the move.
- **The gap to the leaders is a data property, not an unexplored method** (§5, §5a). We enumerate
  **25** architecture/feature/data/calibration levers (FINDINGS §37 for the first 14, §40–§52 for
  the rest): **one** improved the model — the predicted-primary-screen feature (§4) — and four
  *confirmed* existing choices rather than beating them (isoform sharing and separate models per
  track, §7/§8; the uniform interval hinge, §3; and our shipped spread, §6). The search space we
  could reach was explored and characterised, not left open.
- **The last three levers failed with three distinct, identified causes**, which is the useful
  form of a negative result: no ensemble member is both diverse and strong (§8); the metric's
  per-row gradient is already uniform, so no loss reweighting can help (§3); and CYP2D6 is
  data-limited rather than objective-limited — neither a rank transform nor an ordinal
  cumulative-link target moved it (FINDINGS §48, §49a, §51). In each case the *mechanism* was
  confirmed to operate while the gain did not appear, which is why we read the formulation as being
  at the optimum its own assumptions permit.
- **Reproducibility:** pinned data revision, deterministic scaffold folds, `make data | inspect |
  baseline | submit`; experiments in `experiments/` are phased and per-fold/seed checkpointed.
  Neural training is multi-threaded (non-bitwise-reproducible); the ~0.004 seed-ensemble spread
  bounds the noise and every reported delta is judged against it.
- **External data — all public, none merged into a scored column** (each `source × isoform ×
  readout` is a separate head; no cross-assay rescaling):
  - **PubChem AID 1851 (Veith qHTS)** — examined as auxiliary heads; degraded the model; dropped.
  - **ChEMBL / PubChem** — used only to retrieve **structural near-neighbours** of the blinded set
    for an unsupervised **physicochemical-only** warm-start (no assay data, any source); admission
    floor set so admitted compounds are closer to the blind set than our training self-similarity
    (≈0.50). Yielded 324 neighbours; the warm-start was null (§35).
  - **Octant CYP release** (`openadmet/Octant_CYP_inhibition_reactivity_blog_release`, CC-BY-4.0)
    — a pre-ingest **leakage check** read **only** SMILES + identifiers; **assay values were never
    loaded.** It found **5 blinded test compounds present by exact structure and OCNT identifier**
    (+1 near-duplicate at Tanimoto 0.952), and that the release is **99.3% the challenge's own
    CYP3A4 campaign** (1,076 / 1,084) rather than independent data. All 6 overlaps are quarantined
    (`data/octant_quarantine.csv`) from every downstream use; the overlap was **disclosed to the
    organizers** and nothing was built from the release. Octant's `CYP3A4_pIC50` is in any case a
    *different assay condition* (active-enzyme pre-incubation = combined reversible +
    time-dependent), a related but distinct endpoint from the scored `CYP3A4_pIC50_direct_inhibition`.
- **No proprietary data.** Code is open-source (this repository).
