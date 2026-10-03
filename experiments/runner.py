"""Gated phase runner for the four-week experiment plan.

Usage:
    python experiments/runner.py <phase>          # run a phase's steps in order, then stop at its gate
    python experiments/runner.py <phase> --dry    # show what would run / what is already done
    python experiments/runner.py --ledger         # rewrite LEDGER.md from ledger.json and print the tail

Design
------
A *step* is a unit of work that (optionally) runs a shell command and then reports a metrics
dict. Steps are checkpointed by an on-disk artifact (`done_when`), so a kill loses at most the
step in flight; everything already produced is skipped on the next run. Every completed step is
appended to `experiments/ledger.json` and the human-readable `experiments/LEDGER.md` is rewritten
after each one.

A *phase* is an ordered list of steps plus a gate description. The runner NEVER advances past a
phase: it runs that phase's steps, prints the gate report, and exits. Advancing is the operator's
decision.

Long commands are wrapped in `caffeinate -i` (macOS) so an unattended run is not suspended.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LEDGER_JSON = ROOT / "experiments" / "ledger.json"
LEDGER_MD = ROOT / "experiments" / "LEDGER.md"
ISOS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
NF = 5
SEED_FLOOR = 0.004  # deltas below this are noise (3-seed ensemble spread)

# Reference baseline every model step is judged against (D-MPNN + predicted-primary, GFOLD).
BASELINE = {"spearman_macro": 0.6037, "st_rae_macro": 0.4146}


# ----------------------------------------------------------------- ledger ---
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_ledger() -> list[dict]:
    if not LEDGER_JSON.exists():
        return []
    try:
        return json.loads(LEDGER_JSON.read_text())
    except Exception:
        return []


def append_ledger(entry: dict) -> None:
    entries = load_ledger()
    entries.append(entry)
    LEDGER_JSON.write_text(json.dumps(entries, indent=1))
    write_ledger_md(entries)


def completed(phase: str, step: str) -> dict | None:
    """The most recent successful ledger entry for this step, if any."""
    for e in reversed(load_ledger()):
        if e.get("phase") == phase and e.get("step") == step and e.get("status") == "ok":
            return e
    return None


def _fmt_metrics(m: dict) -> str:
    """One-line rendering of whatever a step reported."""
    if not m:
        return "—"
    if "spearman_macro" in m or "st_rae_macro" in m:
        sp, sr = m.get("spearman_macro"), m.get("st_rae_macro")
        bits = []
        if sp is not None:
            bits.append(f"ρ={sp:.4f} ({sp - BASELINE['spearman_macro']:+.4f})")
        if sr is not None:
            bits.append(f"ST-RAE={sr:.4f} ({sr - BASELINE['st_rae_macro']:+.4f})")
        return "  ".join(bits)
    return "  ".join(f"{k}={v}" for k, v in m.items() if not isinstance(v, (dict, list)))


def write_ledger_md(entries: list[dict] | None = None) -> None:
    entries = load_ledger() if entries is None else entries
    lines = [
        "# Experiment ledger",
        "",
        f"Baseline for model steps: macro OOF Spearman **{BASELINE['spearman_macro']}**, "
        f"ST-RAE **{BASELINE['st_rae_macro']}** (D-MPNN + predicted-primary, GFOLD). "
        f"Seed floor ≈ {SEED_FLOOR} — smaller deltas are noise.",
        "",
        "| when (UTC) | phase | step | result | wall | commit | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        wall = e.get("wall_time_s")
        wall_s = f"{wall/60:.1f}m" if isinstance(wall, (int, float)) else "—"
        lines.append(
            f"| {e.get('when','')} | {e.get('phase','')} | `{e.get('step','')}` | "
            f"{_fmt_metrics(e.get('metrics') or {})} | {wall_s} | "
            f"`{e.get('commit','')}` | {e.get('status','')} |"
        )
    # per-isoform detail for model steps
    detail = [e for e in entries if (e.get("metrics") or {}).get("spearman")]
    if detail:
        lines += ["", "## Per-isoform detail", ""]
        for e in detail:
            m = e["metrics"]
            lines.append(f"**{e['phase']} / {e['step']}**  ")
            lines.append("| isoform | Spearman | ST-RAE | ST-RAE fold std |")
            lines.append("|---|---|---|---|")
            for iso in ISOS:
                sp = m.get("spearman", {}).get(iso)
                sr = m.get("st_rae", {}).get(iso)
                fs = m.get("st_rae_fold_std", {}).get(iso)
                lines.append(
                    f"| {iso} | {sp:.3f} | {sr:.3f} | {fs:.3f} |"
                    if None not in (sp, sr, fs) else f"| {iso} | — | — | — |"
                )
            lines.append("")
    LEDGER_MD.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------- standard metrics ---
def _challenge_arrays():
    """Targets, interval bounds, validity mask and GFOLD folds, as used throughout."""
    import pandas as pd

    df = pd.read_csv(ROOT / "data/cyp-challenge-train-test/cyp-challenge-TRAIN_TDI.csv")
    z = np.load(ROOT / "experiments/plog_gfold.npz")
    Y = np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS], 1)
    LO = np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS], 1)
    HI = np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS], 1)
    return Y, LO, HI, ~np.isnan(Y), z["GFOLD"]


def _st_rae(pred, lo, hi, truth) -> float:
    """Soft-threshold relative absolute error (numpy mirror of cyp.losses.st_rae).

    Error is distance outside [lo, hi] (zero inside), relative to the
    predict-the-mean baseline's absolute deviation.
    """
    err = np.maximum(0.0, lo - pred) + np.maximum(0.0, pred - hi)
    denom = max(np.abs(truth - truth.mean()).sum(), 1e-8)
    return float(err.sum() / denom)


def oof_metrics(npy_path: str | Path) -> dict:
    """Standard comparable metrics for an (n_train, 4) OOF prediction array.

    Spearman is pooled over all OOF rows per isoform; ST-RAE is the mean over folds
    (with its fold standard deviation, so a delta can be judged against spread).
    """
    from scipy.stats import spearmanr

    oof = np.load(ROOT / npy_path if not str(npy_path).startswith("/") else npy_path)
    Y, LO, HI, M, GFOLD = _challenge_arrays()
    sp, sr, fstd = {}, {}, {}
    for j, iso in enumerate(ISOS):
        rows = np.where((GFOLD >= 0) & M[:, j] & ~np.isnan(oof[:, j]))[0]
        sp[iso] = float(spearmanr(oof[rows, j], Y[rows, j]).correlation) if len(rows) > 1 else float("nan")
        per_fold = []
        for f in range(NF):
            r = np.where((GFOLD == f) & M[:, j] & ~np.isnan(oof[:, j]))[0]
            if len(r):
                per_fold.append(_st_rae(oof[r, j], LO[r, j], HI[r, j], Y[r, j]))
        sr[iso] = float(np.mean(per_fold)) if per_fold else float("nan")
        fstd[iso] = float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0
    return {
        "spearman": sp,
        "st_rae": sr,
        "st_rae_fold_std": fstd,
        "spearman_macro": float(np.mean([sp[i] for i in ISOS])),
        "st_rae_macro": float(np.mean([sr[i] for i in ISOS])),
    }


# ------------------------------------------------------------- execution ---
def _resolve(dotted: str):
    """Import a dotted path 'module:attr' (module relative to experiments/)."""
    mod_name, _, attr = dotted.partition(":")
    sys.path.insert(0, str(ROOT / "experiments"))
    sys.path.insert(0, str(ROOT / "src"))
    mod = __import__(mod_name, fromlist=["*"])
    return getattr(mod, attr)


def run_command(cmd: str, log: Path, caffeinate: bool = True) -> int:
    """Run a shell command, tee-ing output to `log`. Wrapped in caffeinate -i on macOS."""
    full = cmd
    if caffeinate and Path("/usr/bin/caffeinate").exists():
        full = f"/usr/bin/caffeinate -i {cmd}"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(f"\n$ {full}\n")
        fh.flush()
        proc = subprocess.Popen(
            full, shell=True, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "PYTHONPATH": "src"},
        )
        return proc.wait()


def run_step(phase: str, step: dict) -> dict:
    """Run one step unless already complete. Returns its ledger entry."""
    name = step["name"]
    prior = completed(phase, name)
    done_path = ROOT / step["done_when"] if step.get("done_when") else None
    if prior and (done_path is None or done_path.exists()):
        print(f"  [skip] {name} (done {prior['when']})")
        return prior

    print(f"  [run ] {name}")
    t0 = time.time()
    status, err = "ok", None
    if step.get("run"):
        rc = run_command(step["run"], ROOT / "experiments" / "logs" / f"{phase}_{name}.log")
        if rc != 0:
            status, err = "failed", f"exit {rc}"
    if status == "ok" and done_path is not None and not done_path.exists():
        status, err = "failed", f"missing artifact {step['done_when']}"

    metrics = {}
    if status == "ok" and step.get("metrics"):
        try:
            fn = _resolve(step["metrics"]) if isinstance(step["metrics"], str) else step["metrics"]
            metrics = fn(**step.get("metrics_args", {}))
        except Exception as exc:  # metric failure should not erase the run
            status, err = "failed", f"metrics: {type(exc).__name__}: {exc}"

    entry = {
        "when": _now(), "phase": phase, "step": name, "config": step.get("config", {}),
        "metrics": metrics, "wall_time_s": round(time.time() - t0, 1),
        "commit": _commit_sha(), "status": status,
    }
    if err:
        entry["error"] = err
    append_ledger(entry)
    print(f"  [{status}] {name}  {_fmt_metrics(metrics)}  ({entry['wall_time_s']/60:.1f}m)")
    return entry


def run_phase(phase: str, dry: bool = False) -> None:
    from phases import PHASES  # noqa: E402  (resolved via sys.path in _resolve)

    if phase not in PHASES:
        sys.exit(f"unknown phase '{phase}'. known: {', '.join(PHASES)}")
    spec = PHASES[phase]
    print(f"=== {phase}: {spec.get('title','')} ===")
    if dry:
        for s in spec["steps"]:
            prior = completed(phase, s["name"])
            print(f"  {'done' if prior else 'todo'}  {s['name']}")
        print(f"GATE: {spec.get('gate','')}")
        return

    for s in spec["steps"]:
        entry = run_step(phase, s)
        if entry.get("status") != "ok":
            print(f"\n!! step '{s['name']}' {entry.get('status')}: {entry.get('error')}")
            print("stopping; fix and re-run (completed steps are skipped).")
            return

    print(f"\n=== GATE {phase} ===")
    print(spec.get("gate", ""))
    gate_fn = spec.get("gate_report")
    if gate_fn:
        try:
            print(_resolve(gate_fn)() if isinstance(gate_fn, str) else gate_fn())
        except Exception as exc:
            print(f"(gate report failed: {type(exc).__name__}: {exc})")
    print("\nStopping at the gate. Advancing is the operator's call.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", nargs="?", help="phase to run (see experiments/phases.py)")
    ap.add_argument("--dry", action="store_true", help="show planned/completed steps only")
    ap.add_argument("--ledger", action="store_true", help="rewrite LEDGER.md and print the tail")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "experiments"))
    if args.ledger:
        write_ledger_md()
        entries = load_ledger()
        for e in entries[-10:]:
            print(f"{e['when']}  {e['phase']}/{e['step']}  {_fmt_metrics(e.get('metrics') or {})}  {e['status']}")
        print(f"\n{len(entries)} entries -> {LEDGER_MD.relative_to(ROOT)}")
        return
    if not args.phase:
        ap.error("give a phase, or --ledger")
    run_phase(args.phase, dry=args.dry)


if __name__ == "__main__":
    main()
