"""Phase 1: retrieve unlabelled near neighbours of the 750 blinded compounds (SmallWorld).

Route note. The plan's primary route (download the ~15.26B-SMILES "lots-of-smiles" corpus and
run local ECFP4 search with FAISS/chemfp) needs the HPC cluster: the corpus and its fingerprint
index are far beyond this machine's 24 GB RAM / 526 GiB free disk. This is the authorized
fallback: the SmallWorld API at sw.docking.org against Enamine REAL, batched and resumable.

Only STRUCTURES are retrieved (SMILES + the service's own ECFP4 similarity and molecular
weight). No measured assay label is ever attached to a neighbour — the standing rule holds by
construction, since the service returns none.

    python experiments/sw_retrieve.py --probe   # one query, writes the probe checkpoint
    python experiments/sw_retrieve.py --all     # all 750, per-compound cache, resumable

Per-compound JSON cache under experiments/sw_cache/, so a kill loses at most one query.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "experiments" / "sw_cache"
BLINDED = ROOT / "data/cyp-challenge-train-test/cyp-challenge-TEST-BLINDED.csv"
DIST = 4        # SmallWorld graph-edit distance: close analogues
LENGTH = 200    # hits requested per query (filtered by ECFP4 later, in corpus_filter)
PACE_S = 1.0    # politeness between queries
RETRIES = 3
QUERY_TIMEOUT_S = 150  # hard watchdog: the client has no request timeout, so a hung
                       # socket would otherwise block an unattended run indefinitely


class _QueryTimeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _QueryTimeout()


def _client():
    """SmallWorld client; its init hits /search/maps, which 502s transiently."""
    from smallworld_api import SmallWorld

    last = None
    for attempt in range(5):
        try:
            return SmallWorld()
        except Exception as exc:  # transient proxy errors
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"SmallWorld init failed after retries: {last}")


def _hits(sw, smiles: str) -> list[dict]:
    """Neighbour hits for one query: [{smiles, ecfp4, dist, mw}]. Empty on failure.

    Each attempt is bounded by a SIGALRM watchdog: the service intermittently drops
    connections, and without this a single hung request stalls the whole run.
    """
    for attempt in range(RETRIES):
        try:
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(QUERY_TIMEOUT_S)
            try:
                df = sw.search(smiles, db=sw.REAL_dataset, dist=DIST, length=LENGTH)
            finally:
                signal.alarm(0)
            if df is None or not len(df):
                return []
            col = "hitSmiles" if "hitSmiles" in df.columns else "smiles"
            out = []
            for _, r in df.iterrows():
                s = r.get(col)
                if not isinstance(s, str):
                    continue
                out.append({
                    "smiles": s.split()[0],  # hitSmiles can carry an id suffix
                    "ecfp4": float(r["ecfp4"]) if pd.notna(r.get("ecfp4")) else None,
                    "dist": int(r["dist"]) if pd.notna(r.get("dist")) else None,
                    "mw": float(r["mw"]) if pd.notna(r.get("mw")) else None,
                })
            return out
        except Exception:
            time.sleep(4 * (attempt + 1))
    return []


def run(probe_only: bool = False) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    bl = pd.read_csv(BLINDED)
    sw = _client()
    print(f"SmallWorld ready; db={sw.REAL_dataset} dist={DIST} length={LENGTH}", flush=True)

    rows = list(zip(bl["Molecule_Name"], bl["SMILES"]))
    if probe_only:
        name, smi = rows[0]
        hits = _hits(sw, smi)
        (CACHE / "_probe.json").write_text(json.dumps(
            {"name": name, "smiles": smi, "n_hits": len(hits),
             "db": str(sw.REAL_dataset), "dist": DIST, "length": LENGTH,
             "ecfp4_max": max([h["ecfp4"] for h in hits if h["ecfp4"] is not None], default=None),
             "hits": hits[:25]}, indent=1))
        print(f"probe: {name} -> {len(hits)} hits", flush=True)
        return

    t0, done, failed = time.time(), 0, 0
    for i, (name, smi) in enumerate(rows):
        fp = CACHE / f"{name}.json"
        if fp.exists():
            done += 1
            continue
        hits = _hits(sw, smi)
        # An empty result after retries is cached (with a flag) so the phase can complete;
        # `--retry-empty` clears those so a later pass re-queries them.
        fp.write_text(json.dumps({"name": name, "smiles": smi, "hits": hits,
                                  "empty_after_retries": not hits}))
        done += 1
        if not hits:
            failed += 1
        if i % 25 == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(rows)}] {name}: {len(hits)} hits "
                  f"| elapsed {el/60:.1f}m | empty so far {failed}", flush=True)
        time.sleep(PACE_S)

    cached = len(list(CACHE.glob("OCNT-*.json")))
    if cached >= len(rows):
        (CACHE / "_complete").write_text(f"{cached} queries cached at {time.strftime('%F %T')}\n")
    print(f"done: {cached}/{len(rows)} cached, {failed} empty, {(time.time()-t0)/60:.1f}m", flush=True)


# --------------------------------------------------------------- metrics ---
def probe_metrics() -> dict:
    d = json.loads((CACHE / "_probe.json").read_text())
    return {"probe_hits": d["n_hits"], "probe_ecfp4_max": d.get("ecfp4_max"),
            "db": d["db"], "dist": d["dist"], "length": d["length"]}


def retrieval_metrics() -> dict:
    files = sorted(CACHE.glob("OCNT-*.json"))
    raw, uniq, empty = 0, set(), 0
    for f in files:
        hits = (json.loads(f.read_text()) or {}).get("hits") or []
        if not hits:
            empty += 1
        raw += len(hits)
        uniq.update(h["smiles"] for h in hits if h.get("smiles"))
    return {"queries_cached": len(files), "queries_empty": empty,
            "raw_hits": raw, "unique_hit_smiles": len(uniq)}


def retry_empty() -> int:
    """Delete cache entries that came back empty, so --all re-queries them."""
    n = 0
    for f in CACHE.glob("OCNT-*.json"):
        d = json.loads(f.read_text()) or {}
        if not (d.get("hits") or []):
            f.unlink()
            n += 1
    (CACHE / "_complete").unlink(missing_ok=True)
    print(f"cleared {n} empty cache entries for re-query")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--probe", action="store_true")
    g.add_argument("--all", action="store_true")
    g.add_argument("--retry-empty", action="store_true")
    a = ap.parse_args()
    if a.retry_empty:
        retry_empty()
    else:
        run(probe_only=a.probe)
