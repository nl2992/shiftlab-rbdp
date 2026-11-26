"""RB-DP for a hidden Markov alternative with K states, solved on the reachable belief set (paper Remark 1).

Same programme as Theorem 1 (d = 1): state (t, beta, n, w), beta a belief vector. Instead of a belief grid with
interpolation, every belief reachable from the prior is enumerated (keyed by its rounded value), so the only
discretisation is the w grid {j/N}, N = d_alpha * 2^B, which is exact by Theorem 2 / Corollary 2. Values are
therefore exact up to floating point (no L/U gap).

Also evaluates, by the risk recursion of Lemma 1, the worst-case adaptive risk of the extracted i.i.d.-optimal
policy and of the adaptive-optimal policy.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

INF = float("inf")


@dataclass
class KModel:
    T: int
    B: int
    P: np.ndarray  # K x K row-stochastic
    e: np.ndarray  # P(Y=1 | S=i)
    prior: np.ndarray
    c: float = 0.0
    k: float = 0.0
    alpha: Fraction = Fraction(1, 10)

    def __post_init__(self):
        self.P = np.asarray(self.P, float)
        self.e = np.asarray(self.e, float)
        self.prior = np.asarray(self.prior, float)
        self.N = self.alpha.denominator * 2**self.B
        self.j_alpha = int(self.alpha * self.N)

    def prop(self, b):
        return b @ self.P

    def pos(self, b):
        return float(b @ self.e)

    def post(self, b, y):
        w = b * (self.e if y else 1 - self.e)
        return w / w.sum()


def key(b):
    return tuple(np.round(b, 12))


class Solver:
    def __init__(self, m: KModel, mode: str):
        self.m, self.mode = m, mode
        N = m.N
        i = np.arange(N + 1)[None, :]  # w0 index
        j = np.arange(N + 1)[:, None]  # w' index
        i1 = 2 * j - i
        self.invalid = i1 < 0
        if mode == "causal":
            self.invalid = self.invalid | (i > j)
        self.i1 = np.clip(i1, 0, N)
        self.W, self.dec, self.vpost = {}, {}, {}
        sys.setrecursionlimit(10000)

    # post-promotion optimal cost (unconstrained), action `old` in force
    def post_value(self, t, old, n, b):
        m = self.m
        if t > m.T:
            return 0.0
        kk = (t, old, n, key(b))
        if kk in self.vpost:
            return self.vpost[kk]
        best = min(self._post_choice(t, old, n, b, a, q) for a in (0, 1) for q in ((0, 1) if n else (0,)))
        self.vpost[kk] = best
        return best

    def _post_choice(self, t, old, n, b, a, q):
        m = self.m
        p = m.pos(b)
        v = (p if a == 0 else 1 - p) + m.c * q + m.k * (a != old)
        if q:
            v += (1 - p) * self.post_value(t + 1, a, n - 1, m.prop(m.post(b, 0))) + p * self.post_value(
                t + 1, a, n - 1, m.prop(m.post(b, 1)))
        else:
            v += self.post_value(t + 1, a, n, m.prop(b))
        return v

    def promote_cost(self, t, n, b):
        return min(self._post_choice(t, 0, n, b, 1, q) for q in ((0, 1) if n else (0,)))

    def value(self, t, b, n):
        """W(t, b, n, .) on the grid j/N, j = 0..N."""
        m = self.m
        if t > m.T:
            return np.zeros(m.N + 1)
        kk = (t, key(b), n)
        if kk in self.W:
            return self.W[kk]
        p = m.pos(b)
        CN = p + self.value(t + 1, m.prop(b), n)
        branch = np.zeros(m.N + 1, dtype=np.int8)  # 0 = no query, 1 = query, 2 = promote
        arg = np.zeros(m.N + 1, dtype=np.int32)
        f = CN.copy()
        if n:
            g0 = (1 - p) * self.value(t + 1, m.prop(m.post(b, 0)), n - 1)
            g1 = p * self.value(t + 1, m.prop(m.post(b, 1)), n - 1)
            M = g0[None, :] + g1[self.i1]
            M[self.invalid] = INF
            arg = M.argmin(axis=1).astype(np.int32)
            CQ = p + m.c + M[np.arange(m.N + 1), arg]
            q = CQ < CN - 1e-13
            f = np.where(q, CQ, CN)
            branch[q] = 1
        P = self.promote_cost(t, n, b)
        if P < f[-1] - 1e-13:
            f[-1], branch[-1] = P, 2
        W, verts = lower_hull(f)
        self.W[kk] = W
        self.dec[kk] = (verts, branch, arg)
        return W

    def risk(self, t, b, n, j, null, memo):
        """Worst-case risk under `null` ('causal' or 'iid') of this solver's policy started at w = j/N."""
        m = self.m
        if t > m.T:
            return 0.0
        kk = (t, key(b), n, j)
        if kk in memo:
            return memo[kk]
        self.value(t, b, n)
        verts, branch, arg = self.dec[(t, key(b), n)]
        pos = np.searchsorted(verts, j)
        if verts[pos] == j:
            pts = [(j, 1.0)]
        else:
            lo, hi = verts[pos - 1], verts[pos]
            lam = (hi - j) / (hi - lo)
            pts = [(lo, lam), (hi, 1 - lam)]
        r = 0.0
        for x, wt in pts:
            br = branch[x]
            if br == 2:
                rx = 1.0
            elif br == 0:
                rx = self.risk(t + 1, m.prop(b), n, x, null, memo)
            else:
                i0 = int(arg[x])
                i1 = min(2 * x - i0, m.N)
                r0 = self.risk(t + 1, m.prop(m.post(b, 0)), n - 1, i0, null, memo)
                r1 = self.risk(t + 1, m.prop(m.post(b, 1)), n - 1, i1, null, memo)
                rx = max(r0, (r0 + r1) / 2) if null == "causal" else (r0 + r1) / 2
            r += wt * rx
        memo[kk] = r
        return r


def lower_hull(f):
    """Lower convex hull of points (j, f[j]); returns hull values on the grid and the vertex indices."""
    h = []
    for x in range(len(f)):
        while len(h) >= 2:
            x1, x2 = h[-2], h[-1]
            if (f[x2] - f[x1]) * (x - x1) >= (f[x] - f[x1]) * (x2 - x1) - 1e-15:
                h.pop()
            else:
                break
        h.append(x)
    verts = np.array(h)
    return np.interp(np.arange(len(f)), verts, f[verts]), verts


def solve(m: KModel) -> dict:
    out = {}
    b1 = m.prop(m.prior)
    for mode in ("causal", "iid"):
        s = Solver(m, mode)
        V = s.value(1, b1, m.B)[m.j_alpha]
        out[f"V_{mode}"] = float(V)
        out[f"risk_causal_{mode}pol"] = s.risk(1, b1, m.B, m.j_alpha, "causal", {})
        out[f"risk_iid_{mode}pol"] = s.risk(1, b1, m.B, m.j_alpha, "iid", {})
        out[f"states_{mode}"] = len(s.W)
    b, C0 = b1, 0.0
    for _ in range(m.T):
        C0 += m.pos(b)
        b = m.prop(b)
    out["C0"] = C0
    out["Gamma"] = out["V_causal"] - out["V_iid"]
    voi = C0 - out["V_iid"]
    out["VOI_iid"] = voi
    out["rho"] = out["Gamma"] / voi if voi > 1e-12 else 0.0
    return out
