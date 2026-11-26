"""Three-state premium experiment (Section 5): cyclic regime chain low -> mid -> high -> low.

P(S_{t+1} = S_t + 1 mod 3) = eta, P(stay) = 1 - eta; challenger win probabilities e = (1/10, 1/2, 9/10);
stationary (uniform) prior; alpha = 1/10, c = k = 0, d = 1. Solved exactly on the reachable belief set.
Run from the repo root:
    uv run --no-sync python scripts/paper4_continuation/rbdp/kstate/run_kstate_premium.py
"""
from __future__ import annotations

import json
import sys
import time
from fractions import Fraction as F
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rbdp_kstate import KModel, solve  # noqa: E402

OUT = HERE.parents[3] / "docs/paper4_continuation/rbdp/results/kstate_premium_v1.jsonl"
ETAS = [F(2, 3), F(3, 4), F(9, 10)]
E = [0.1, 0.5, 0.9]
TS = [4, 8, 12, 16, 24, 32]
BS = [1, 2, 3]


def cyclic(eta: F):
    a, s = float(eta), float(1 - eta)
    return [[s, a, 0.0], [0.0, s, a], [a, 0.0, s]]


def job(args):
    eta, B, T = args
    t0 = time.time()
    m = KModel(T=T, B=B, P=cyclic(eta), e=E, prior=[1 / 3] * 3, alpha=F(1, 10))
    o = solve(m)
    o.update(eta=str(eta), B=B, T=T, K=3, e=E, alpha="1/10", c=0, k=0, secs=time.time() - t0)
    return o


if __name__ == "__main__":
    jobs = [(eta, B, T) for eta in ETAS for B in BS for T in TS]
    jobs.sort(key=lambda j: -(j[2] * 2 ** j[1]))
    with Pool(4) as pool, open(OUT, "w") as fh:
        for o in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(o) + "\n")
            fh.flush()
            print(f"eta={o['eta']:>5} B={o['B']} T={o['T']:>2} rho={o['rho']:.4f} "
                  f"iidpol_causal_risk={o['risk_causal_iidpol']:.3f} causalpol_risk={o['risk_causal_causalpol']:.3f} "
                  f"{o['secs']:.1f}s", flush=True)
