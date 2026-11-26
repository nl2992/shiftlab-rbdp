"""Generic risk-threshold DP baseline: the same risk-budget Bellman recursion on a uniform risk grid {j/M}.

This is the standard way to make a continuous risk-to-go state finite (risk-threshold dynamic programming with a
discretised threshold). Controls are restricted to the grid, so the extracted policy is feasible and its exactly
evaluated cost U_M is an upper bound on V*. RB-DP uses the dyadic lattice N = d_alpha * 2^B (Theorem 2), on which
the restriction loses nothing. Both use the same belief grid (G = 401) and the same exact policy evaluation, so
U_M - U_RBDP is the cost of the grid choice alone. Adaptive (causal) null, d = 1.

Run from the repo root:
    uv run --no-sync python scripts/paper4_continuation/rbdp/baselines/run_uniform_grid.py
"""
from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

from shiftlab.paper4.risk_budget_dp_fast import RBDPConfig, solve

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs/paper4_continuation/rbdp/results/uniform_grid_baseline_v1.jsonl"
ETAS = [2 / 3, 3 / 4, 9 / 10]
BS = [2, 3, 4, 5, 6]
MS = [30, 50, 100, 200, 500, 1000]
T, E, ALPHA = 32, 0.1, 0.1


def job(args):
    eta, B, M = args
    base = dict(T=T, B=B, t01=eta, t10=eta, e0=E, e1=1 - E, prior1=0.5, alpha=ALPHA, mode="causal", n_beta=401)
    if M is None:
        cfg = RBDPConfig(**base, lattice_exact=True)
    else:
        cfg = RBDPConfig(**base, lattice_exact=False, w_step_div=round(M * ALPHA), compute_lower=False)
    t0 = time.time()
    r = solve(cfg)
    secs = time.time() - t0
    return {"eta": eta, "B": B, "M": M if M is not None else cfg.n_w, "rbdp": M is None, "N_w": cfg.n_w,
            "U": r["U"], "L": r.get("L"), "C0": r["C0"], "risk_causal": r["risk_causal"], "secs": secs, "T": T, "e": E,
            "alpha": ALPHA}


if __name__ == "__main__":
    jobs = [(eta, B, M) for eta in ETAS for B in BS for M in [None] + MS]
    jobs.sort(key=lambda j: -(2 ** j[1] if j[2] is None else j[2]))
    with Pool(4) as pool, open(OUT, "w") as fh:
        for o in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(o) + "\n")
            fh.flush()
            print(f"eta={o['eta']:.3f} B={o['B']} {'RB-DP' if o['rbdp'] else 'grid'} N={o['N_w']:>5} "
                  f"U={o['U']:.6f} risk={o['risk_causal']:.4f} {o['secs']:.1f}s", flush=True)
