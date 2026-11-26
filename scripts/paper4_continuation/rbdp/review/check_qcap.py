"""Check (q): Theorems 1-2 for a rational null cap p_t in [0, q] and a K-state alternative.

Claims checked (q = a/b in (0, 1), d = 1, any finite number K of hidden states):
  * Theorem 1 with m_q(w0, w1) = max{w0, (1-q) w0 + q w1} (adaptive) and (1-q) w0 + q w1 (i.i.d. null p = q):
    the risk-budget value equals the exact optimum from an LP over the learner's full sequence-form tree
    (adaptive null as the sequence-form dual of a null that plays p in {0, q} at each query node).
  * Theorem 2: every breakpoint of W(t, beta, n, .) lies in L_n, where L_0 = {0, 1} and
    L_n = {(1-q) x + q y : x, y in L_{n-1}}  (so L_n is in b^{-n} Z; for q = 1/2, L_n = D_n).
    Checked by exact hull vertices and by chord tests on every cell between consecutive points of L_n.

Exact rationals throughout except the LP (HiGHS, float). Run from the repo root:
    uv run --no-sync python scripts/paper4_continuation/rbdp/review/check_qcap.py
"""
from __future__ import annotations

import bisect
import json
import random
import sys
import time
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_b_lp import Tree, solve_lp  # noqa: E402
from check_c_lattice import ev, lower_hull  # noqa: E402
from check_kstate import KCell, rand_row  # noqa: E402
from common import post_value_fn  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402

OUT = HERE.parents[3] / "docs/paper4_continuation/rbdp/results/review/check_qcap_results.json"


# ---------------------------------------------------------------- lattice L_n
def lattice(q: F, n: int) -> list:
    L = {F(0), F(1)}
    for _ in range(n):
        L = {(1 - q) * x + q * y for x in L for y in L}
    return sorted(L)


# ---------------------------------------------------------------- exact risk-budget DP with cap q
def run(cell, q: F, mode: str, checks: dict):
    Vpost, P = post_value_fn(cell)
    Ls = {}

    def Lat(n):
        if n not in Ls:
            Ls[n] = lattice(q, n)
        return Ls[n]

    @lru_cache(maxsize=None)
    def W(t, b, n):
        if t > cell.T:
            return ((F(0), F(0)), (F(1), F(0)))
        p = cell.pos(b)
        CN = [(x, y + p) for x, y in W(t + 1, cell.prop(b), n)]
        Ln = Lat(n)
        CQ = None
        if n:
            g = []
            for y, py in ((0, 1 - p), (1, p)):
                Wy = W(t + 1, cell.prop(cell.post(b, y)), n - 1)
                g.append([(x, py * v) for x, v in Wy])
            X0, X1 = [x for x, _ in g[0]], [x for x, _ in g[1]]

            def cq(w):
                best = None
                A = set(X0) | {F(0), F(1)} | ({w} if mode == "causal" else set())
                Bs = set(X1) | {F(0), F(1)}
                cand = {(a, bb) for a in A for bb in Bs}
                cand |= {(a, (w - (1 - q) * a) / q) for a in A}
                cand |= {((w - q * bb) / (1 - q), bb) for bb in Bs}
                for a, bb in cand:
                    if not (0 <= a <= 1 and 0 <= bb <= 1 and (1 - q) * a + q * bb <= w):
                        continue
                    if mode == "causal" and a > w:
                        continue
                    v = ev(g[0], a) + ev(g[1], bb)
                    if best is None or v < best:
                        best = v
                return best + p + cell.c

            vals = {x: cq(x) for x in Ln}
            for lo, hi in zip(Ln, Ln[1:]):
                for fr in (F(1, 3), F(1, 2), F(2, 3)):
                    x = lo + fr * (hi - lo)
                    checks["chord_tests"] += 1
                    if cq(x) != vals[lo] + fr * (vals[hi] - vals[lo]):
                        checks["chord_fail"].append((t, n, str(x)))
            CQ = [(x, vals[x]) for x in Ln]
        xs = sorted(set(x for x, _ in CN) | set(Ln))
        pts = []
        for i, x in enumerate(xs):
            yN = ev(CN, x)
            y = yN if CQ is None else min(yN, ev(CQ, x))
            if x == 1:
                y = min(y, P(t, n, b))
            pts.append((x, y))
            if CQ is not None and i + 1 < len(xs):
                x2 = xs[i + 1]
                d1, d2 = ev(CN, x) - ev(CQ, x), ev(CN, x2) - ev(CQ, x2)
                if d1 * d2 < 0:
                    xc = x + (x2 - x) * d1 / (d1 - d2)
                    pts.append((xc, ev(CN, xc)))
        H = lower_hull(pts)
        checks["rows"] += 1
        Lset = set(Ln)
        for x, _ in H:
            if x not in Lset:
                checks["offlattice"].append((t, n, str(x)))
        return tuple(H)

    root = W(1, cell.beta1, cell.B)
    return ev(root, cell.alpha)


# ---------------------------------------------------------------- independent LP with cap q
def risk_coeff(tree: Tree, pfun):
    coef = {}
    for s, labels in tree.first_promo:
        pr = F(1)
        qt = dict(labels)
        for (t, y) in labels:
            p = pfun(tuple(qt.get(u) for u in range(1, t)))
            pr *= p if y else 1 - p
            if pr == 0:
                break
        if pr:
            coef[s] = coef.get(s, 0) + float(pr)
    return coef


def null_sf(tree: Tree, cell, q: F):
    """Null sequence form with actions p in {0, q} at each query node (mixtures give all of [0, q])."""
    seqs, infos, count = {}, [], 1

    def walk(steps, ptuple, par):
        nonlocal count
        t = len(steps) + 1
        if t > cell.T:
            return
        nq = sum(1 for s in steps if s is not None)
        walk(steps + (None,), ptuple, par)
        if nq < cell.B:
            kids = []
            for p in (F(0), q):
                seqs[(steps, ptuple + (p,))] = count
                kids.append(count)
                count += 1
            infos.append((par, kids))
            for p, k in zip((F(0), q), kids):
                for y in ((0,) if p == 0 else (0, 1)):
                    walk(steps + (y,), ptuple + (p,), k)

    walk((), (), 0)
    ny = count
    rows, cols, vals = [0], [0], [1.0]
    for i, (par, kids) in enumerate(infos, start=1):
        rows += [i] * 3
        cols += [par] + kids
        vals += [-1.0, 1.0, 1.0]
    Fm = coo_matrix((vals, (rows, cols)), shape=(len(infos) + 1, ny)).tocsr()
    ar, ac, av = [], [], []
    for s, labels in tree.first_promo:
        qt = dict(labels)
        if not labels:
            ar.append(s); ac.append(0); av.append(1.0)
            continue
        Ln = len(labels)
        for bits in range(2**Ln):
            ps = tuple(q if (bits >> i) & 1 else F(0) for i in range(Ln))
            pr = F(1)
            for (t, y), p in zip(labels, ps):
                pr *= p if y else 1 - p
            if pr == 0:
                continue
            tlast, _ = labels[-1]
            key = (tuple(qt.get(u) for u in range(1, tlast)), ps)
            ar.append(s); ac.append(seqs[key]); av.append(float(pr))
    A = coo_matrix((av, (ar, ac)), shape=(tree.nseq, ny)).tocsr()
    return A, Fm, ny


def lp_values(cell, q: F):
    tree = Tree(cell)
    v_iid, _ = solve_lp(tree, [risk_coeff(tree, lambda st: q)], cell.alpha)
    v_c, _ = solve_lp(tree, None, cell.alpha, dual=null_sf(tree, cell, q))
    return v_c, v_iid


# ---------------------------------------------------------------- driver
def rand_cell(rng):
    K = rng.choice([2, 3, 3, 4])
    if rng.random() < 0.5:  # cyclic, prone to a premium
        st = F(rng.randint(1, 4), 20)
        P = tuple(tuple((1 - st) if j == (i + 1) % K else st if j == i else F(0) for j in range(K)) for i in range(K))
    else:
        P = tuple(rand_row(rng, K) for _ in range(K))
    e = tuple(sorted(F(rng.randint(1, 19), 20) for _ in range(K)))
    if len(set(e)) < K:
        return rand_cell(rng)
    cell = KCell(T=rng.randint(3, 5), B=rng.randint(1, 3), P=P, e=e, prior=tuple(F(1, K) for _ in range(K)),
                 c=rng.choice([F(0), F(1, 100)]), k=rng.choice([F(0), F(1, 20)]),
                 alpha=rng.choice([F(1, 10), F(1, 7), F(1, 5), F(1, 4)]))
    q = rng.choice([F(1, 3), F(2, 5), F(1, 2), F(3, 5), F(2, 3), F(3, 4)])
    return cell, q


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    rng = random.Random(20261128)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    res, t0 = [], time.time()
    for i in range(n):
        cell, q = rand_cell(rng)
        rec = {"cell": cell.describe(), "q": str(q)}
        v_c_lp, v_i_lp = lp_values(cell, q)
        for mode, vlp in (("causal", v_c_lp), ("iid", v_i_lp)):
            ch = {"rows": 0, "chord_tests": 0, "chord_fail": [], "offlattice": []}
            v = run(cell, q, mode, ch)
            rec[mode] = {"V_dp": float(v), "V_lp": vlp, "diff": float(v) - vlp, "rows": ch["rows"],
                         "chord_tests": ch["chord_tests"], "n_chord_fail": len(ch["chord_fail"]),
                         "n_offlattice": len(ch["offlattice"]), "examples": (ch["chord_fail"] + ch["offlattice"])[:3]}
        rec["gamma"] = rec["causal"]["V_dp"] - rec["iid"]["V_dp"]
        res.append(rec)
        print(f"{i:2d} K={cell.K} q={q} T={cell.T} B={cell.B} dC={rec['causal']['diff']:+.1e} "
              f"dI={rec['iid']['diff']:+.1e} gamma={rec['gamma']:.2e} "
              f"off={rec['causal']['n_offlattice'] + rec['iid']['n_offlattice']} "
              f"chordfail={rec['causal']['n_chord_fail'] + rec['iid']['n_chord_fail']}", flush=True)
    summ = {"cells": n, "q_values": sorted({r["q"] for r in res}), "K_values": sorted({r["cell"]["K"] for r in res}),
            "max_abs_diff": max(max(abs(r["causal"]["diff"]), abs(r["iid"]["diff"])) for r in res),
            "cells_positive_premium": sum(r["gamma"] > 1e-9 for r in res),
            "rows": sum(r["causal"]["rows"] + r["iid"]["rows"] for r in res),
            "chord_tests": sum(r["causal"]["chord_tests"] + r["iid"]["chord_tests"] for r in res),
            "chord_fail": sum(r["causal"]["n_chord_fail"] + r["iid"]["n_chord_fail"] for r in res),
            "offlattice": sum(r["causal"]["n_offlattice"] + r["iid"]["n_offlattice"] for r in res),
            "seed": 20261128, "secs": time.time() - t0}
    print(json.dumps(summ, indent=1))
    OUT.write_text(json.dumps({"summary": summ, "cells": res}, indent=1, default=str))
