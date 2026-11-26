"""Uniform-grid risk-threshold DP vs RB-DP on random cells (asymmetric chains, label and switching costs, several alpha).

Cells are drawn like the Corollary 2 stress test: transitions and emissions in twentieths (half anti-persistent),
prior in {0, 1/3, 1/2}, c in {0, 1/100}, k in {0, 1/20}, alpha in {1/10, 1/7, 3/20, 1/20}; T in {12, 16, 24},
B in {2, ..., 5}. For each cell: RB-DP (dyadic lattice, L and U) and the uniform grid at M = round(K / alpha)
cells for K in {3, 5, 10, 20}, with its own rounded lower bound L_M and policy cost U_M. Adaptive null, d = 1.

Run from the repo root:
    uv run --no-sync python scripts/paper4_continuation/rbdp/baselines/run_uniform_grid_random.py
"""
from __future__ import annotations

import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

from shiftlab.paper4.risk_budget_dp_fast import RBDPConfig, solve

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs/paper4_continuation/rbdp/results/uniform_grid_baseline_random_v1.jsonl"
KS = [3, 5, 10, 20]
N_CELLS = 40


def cells():
    rng = random.Random(20261127)
    out = []
    while len(out) < N_CELLS:
        anti = rng.random() < 0.5
        lo = 11 if anti else 1
        t01, t10 = rng.randint(lo, 19) / 20, rng.randint(lo, 19) / 20
        e0, e1 = rng.randint(1, 9) / 20, rng.randint(11, 19) / 20
        out.append(dict(T=rng.choice([12, 16, 24]), B=rng.randint(2, 5), t01=t01, t10=t10, e0=e0, e1=e1,
                        prior1=rng.choice([0.0, 1 / 3, 0.5]), c=rng.choice([0.0, 0.01]), k=rng.choice([0.0, 0.05]),
                        alpha=rng.choice([0.1, 1 / 7, 0.15, 0.05])))
    return out


def job(args):
    i, cell, K = args
    base = dict(**cell, mode="causal", n_beta=401)
    cfg = RBDPConfig(**base, lattice_exact=True) if K is None else RBDPConfig(**base, lattice_exact=False, w_step_div=K)
    t0 = time.time()
    r = solve(cfg)
    return {"cell_id": i, "cell": cell, "K": K, "rbdp": K is None, "N_w": cfg.n_w, "U": r["U"], "L": r["L"],
            "C0": r["C0"], "risk_causal": r["risk_causal"], "secs": time.time() - t0}


if __name__ == "__main__":
    cs = cells()
    jobs = [(i, c, K) for i, c in enumerate(cs) for K in [None] + KS]
    with Pool(4) as pool, open(OUT, "w") as fh:
        for o in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(o) + "\n")
            fh.flush()
