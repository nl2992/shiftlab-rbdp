"""Check (k): Theorems 1 and 2 for alternatives with K > 2 hidden states (paper Remark 1).

The two existing review checks are reused unchanged; only the alternative model is swapped for a K-state
hidden Markov chain with the same interface (prop, pos, post, beta1), with beliefs as exact tuples:

  * check_b_lp.exact_values: exact optimum by LP over the learner's full sequence-form tree (no belief or
    risk-budget state), for the i.i.d. null and for the adaptive null (dual form; also explicit
    enumeration of reduced nulls when there are at most 40,000).
  * check_c_lattice.run: the risk-budget programme of Theorem 1 at exact beliefs in exact rationals, with
    a record of every hull vertex off the dyadic lattice D_n and every failed chord test (Theorem 2).

Theorem 1 predicts LP value == risk-budget value (both modes); Theorem 2 predicts no off-lattice vertex
and no failed chord test. Run from the repo root:
    uv run --no-sync python scripts/paper4_continuation/rbdp/review/check_kstate.py 40
    uv run --no-sync python scripts/paper4_continuation/rbdp/review/check_kstate.py 30 cyclic
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_b_lp import exact_values  # noqa: E402
from check_c_lattice import run  # noqa: E402

OUT = HERE.parents[3] / "docs/paper4_continuation/rbdp/results/review/check_kstate_results.json"


@dataclass(frozen=True)
class KCell:
    T: int
    B: int
    P: tuple  # row-stochastic K x K transition matrix, P[i][j] = P(S_{t+1}=j | S_t=i)
    e: tuple  # e[i] = P(Y=1 | S=i), all in (0, 1)
    prior: tuple
    c: F = F(0)
    k: F = F(0)
    alpha: F = F(1, 10)

    @property
    def K(self) -> int:
        return len(self.e)

    def prop(self, b):
        return tuple(sum(b[i] * self.P[i][j] for i in range(self.K)) for j in range(self.K))

    def pos(self, b):
        return sum(bi * ei for bi, ei in zip(b, self.e))

    def post(self, b, y):
        w = [bi * (ei if y else 1 - ei) for bi, ei in zip(b, self.e)]
        z = sum(w)
        return tuple(x / z for x in w)

    @property
    def beta1(self):
        return self.prop(self.prior)

    def describe(self):
        s = lambda x: str(x)  # noqa: E731
        return {"T": self.T, "B": self.B, "K": self.K, "P": [[s(x) for x in r] for r in self.P],
                "e": [s(x) for x in self.e], "prior": [s(x) for x in self.prior], "c": s(self.c),
                "k": s(self.k), "alpha": s(self.alpha)}


def rand_row(rng, K, d=20):
    cuts = sorted(rng.randint(1, d - 1) for _ in range(K - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [d])]
    if any(p == 0 for p in parts):
        return rand_row(rng, K, d)
    return tuple(F(p, d) for p in parts)


def rand_cell(rng, K):
    """Half the cells use a strong cyclic chain (state i -> i+1), which tends to give a positive premium."""
    cyclic = rng.random() < 0.5
    if cyclic:
        stay = F(rng.randint(1, 3), 20)
        back = F(rng.randint(1, 3), 20)
        P = tuple(tuple((1 - stay - back) if j == (i + 1) % K else stay if j == i else back
                        if j == (i - 1) % K else F(0) for j in range(K)) for i in range(K))
    else:
        P = tuple(rand_row(rng, K) for _ in range(K))
    e = tuple(sorted(F(rng.randint(1, 19), 20) for _ in range(K)))
    if len(set(e)) < K:
        return rand_cell(rng, K)
    prior = rng.choice([tuple(F(1, K) for _ in range(K)), rand_row(rng, K, 12)])
    return KCell(T=rng.randint(3, 5), B=rng.randint(1, 3), P=P, e=e, prior=prior,
                 c=rng.choice([F(0), F(1, 100), F(1, 30)]), k=rng.choice([F(0), F(1, 20)]),
                 alpha=rng.choice([F(1, 10), F(1, 7), F(3, 20), F(1, 5), F(1, 3)]), ), cyclic


def rand_cyclic_cell(rng, K):
    """Targeted batch: strong cyclic chains, longer horizons, no costs (where a premium is likely)."""
    stay, back = F(rng.randint(0, 2), 20), F(rng.randint(0, 2), 20)
    P = tuple(tuple((1 - stay - back) if j == (i + 1) % K else stay if j == i else back
                    if j == (i - 1) % K else F(0) for j in range(K)) for i in range(K))
    e = tuple(sorted(F(rng.randint(1, 19), 20) for _ in range(K)))
    if len(set(e)) < K:
        return rand_cyclic_cell(rng, K)
    return KCell(T=rng.randint(5, 6), B=rng.randint(2, 3), P=P, e=e, prior=tuple(F(1, K) for _ in range(K)),
                 alpha=rng.choice([F(1, 10), F(1, 7), F(1, 5)])), True


def check(cell: KCell):
    assert all(sum(r) == 1 for r in cell.P) and sum(cell.prior) == 1
    rec = {"cell": cell.describe()}
    ex = exact_values(cell)
    rec["lp"] = {k: (float(v) if isinstance(v, (float, F)) else v) for k, v in ex.items()}
    for mode, key in (("causal", "V_causal_dual"), ("iid", "V_iid")):
        checks = {"rows": 0, "chord_tests": 0, "chord_fail": [], "offlattice_vertices": [],
                  "crossings": 0, "increasing": 0}
        v, _ = run(cell, mode, checks)
        rec[mode] = {"V_rbdp_exact": str(v), "V_rbdp": float(v), "V_lp": ex[key],
                     "diff": float(v) - ex[key], "rows": checks["rows"],
                     "chord_tests": checks["chord_tests"], "chord_fail": checks["chord_fail"][:5],
                     "n_chord_fail": len(checks["chord_fail"]),
                     "offlattice": checks["offlattice_vertices"][:5],
                     "n_offlattice": len(checks["offlattice_vertices"]), "increasing": checks["increasing"]}
    if "V_causal_enum" in ex:
        rec["enum_minus_dual"] = ex["V_causal_enum"] - ex["V_causal_dual"]
    rec["gamma"] = rec["causal"]["V_rbdp"] - rec["iid"]["V_rbdp"]
    return rec


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    targeted = len(sys.argv) > 2 and sys.argv[2] == "cyclic"
    seed = 20261126 if targeted else 20261125
    rng = random.Random(seed)
    n_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    if targeted:
        OUT = OUT.with_name("check_kstate_cyclic_results.json")
    results, t0 = [], time.time()
    for i in range(n_cells):
        K = 3 if i < int(0.75 * n_cells) else 4
        cell, cyclic = (rand_cyclic_cell if targeted else rand_cell)(rng, K)
        t1 = time.time()
        rec = check(cell)
        rec["cyclic"], rec["secs"] = cyclic, time.time() - t1
        results.append(rec)
        print(f"{i:2d} K={K} T={cell.T} B={cell.B} cyc={int(cyclic)} "
              f"dC={rec['causal']['diff']:+.1e} dI={rec['iid']['diff']:+.1e} gamma={rec['gamma']:.2e} "
              f"off={rec['causal']['n_offlattice'] + rec['iid']['n_offlattice']} "
              f"chordfail={rec['causal']['n_chord_fail'] + rec['iid']['n_chord_fail']} "
              f"rows={rec['causal']['rows'] + rec['iid']['rows']} {rec['secs']:.1f}s", flush=True)
    summ = {
        "cells": len(results),
        "K_counts": {str(K): sum(r["cell"]["K"] == K for r in results) for K in (3, 4)},
        "max_abs_diff_causal": max(abs(r["causal"]["diff"]) for r in results),
        "max_abs_diff_iid": max(abs(r["iid"]["diff"]) for r in results),
        "max_abs_enum_minus_dual": max((abs(r["enum_minus_dual"]) for r in results if "enum_minus_dual" in r),
                                       default=None),
        "cells_with_enum": sum("enum_minus_dual" in r for r in results),
        "cells_positive_premium": sum(r["gamma"] > 1e-9 for r in results),
        "value_rows_checked": sum(r["causal"]["rows"] + r["iid"]["rows"] for r in results),
        "chord_tests": sum(r["causal"]["chord_tests"] + r["iid"]["chord_tests"] for r in results),
        "chord_fail": sum(r["causal"]["n_chord_fail"] + r["iid"]["n_chord_fail"] for r in results),
        "offlattice_vertices": sum(r["causal"]["n_offlattice"] + r["iid"]["n_offlattice"] for r in results),
        "seed": seed, "secs": time.time() - t0,
    }
    print(json.dumps(summ, indent=1))
    OUT.write_text(json.dumps({"summary": summ, "cells": results}, indent=1, default=str))
