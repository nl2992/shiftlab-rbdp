"""Extension of the multi-state premium experiment: longer horizons, B = 4, and a four-state cyclic chain.

Same family as run_kstate_premium.py: P(advance one regime) = eta, P(stay) = 1 - eta, uniform prior,
alpha = 1/10, c = k = 0, d = 1, solved exactly on the reachable belief set.
  K = 3, e = (1/10, 1/2, 9/10): T in {48, 64} for B <= 2, T = 100 for B = 1, T = 48 for B = 3,
                                B = 4 for T in {8, 12, 16, 20}.
  K = 4, e = (1/10, 7/20, 13/20, 9/10): B <= 3, T in {8, 16, 24, 32}.
Run from the repo root:
    uv run --no-sync python scripts/paper4_continuation/rbdp/kstate/run_kstate_extend.py
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

OUT = HERE.parents[3] / "docs/paper4_continuation/rbdp/results/kstate_premium_v2.jsonl"
ETAS = [F(2, 3), F(3, 4), F(9, 10)]
EMIT = {3: [0.1, 0.5, 0.9], 4: [0.1, 0.35, 0.65, 0.9]}


def cyclic(eta: F, K: int):
    a, s = float(eta), float(1 - eta)
    return [[a if j == (i + 1) % K else s if j == i else 0.0 for j in range(K)] for i in range(K)]


def job(args):
    K, eta, B, T = args
    t0 = time.time()
    m = KModel(T=T, B=B, P=cyclic(eta, K), e=EMIT[K], prior=[1 / K] * K, alpha=F(1, 10))
    o = solve(m)
    o.update(eta=str(eta), B=B, T=T, K=K, e=EMIT[K], alpha="1/10", c=0, k=0, secs=time.time() - t0)
    return o


if __name__ == "__main__":
    jobs = []
    for eta in ETAS:
        jobs += [(3, eta, B, T) for B in (1, 2) for T in (48, 64)]
        jobs += [(3, eta, 1, 100), (3, eta, 3, 48)]
        jobs += [(3, eta, 4, T) for T in (8, 12, 16, 20)]
        jobs += [(4, eta, B, T) for B in (1, 2, 3) for T in (8, 16, 24, 32)]
    jobs.sort(key=lambda j: -(j[3] ** (j[2] + 1)))
    with Pool(5) as pool, open(OUT, "w") as fh:
        for o in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(o) + "\n")
            fh.flush()
            print(f"K={o['K']} eta={o['eta']:>5} B={o['B']} T={o['T']:>3} rho={o['rho']:.4f} "
                  f"iidpol={o['risk_causal_iidpol']:.3f} causalpol={o['risk_causal_causalpol']:.3f} "
                  f"{o['secs']:.1f}s", flush=True)
