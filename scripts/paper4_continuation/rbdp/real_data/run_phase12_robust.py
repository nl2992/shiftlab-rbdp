"""Phase 12b: replay robustness over label budget B, horizon T, switching cost k and alternative misspecification.

Same protocol as Phase 9 replay (fit on first half of episodes, replay on non-overlapping T-windows of the second half,
alpha = 0.1, c = 0.01) with one factor changed per configuration. Resume-safe; one jsonl line per (config, dataset).
Usage: run_phase12_robust.py run CONFIG [--workers=N] [--subset=N] ; run_phase12_robust.py time CONFIG ; summary
"""
import json, math, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_phase9 as p9  # noqa: E402
p8 = p9.p8
RES = p8.RES
OUT = RES / "phase12b_robust.jsonl"
POLS = ["P5", "P7", "P5r", "never"]  # stress tests: solver-based policies only (heuristic calibrations are too slow at large B, T)

class SetupLite(p9.Setup9):
    """Setup9 without the iidlib heuristic baselines (P1-P4, P3c, P4c, P6): chain, posterior, P5/P7 solves, P5r plan."""
    def __init__(self, sp, T, B, alpha, c, k):
        from iidlib import Chain, Post
        from shiftlab.paper4.risk_budget_dp_fast import RBDPConfig, _Evaluator, _Solver
        import sys as _s
        self.ch = ch = Chain(sp["t01"], sp["t10"], sp["e0"], sp["e1"], sp["prior1"], T, B, c, k)
        self.alpha, self.T, self.B, self.c, self.k = alpha, T, B, c, k
        self.post = Post(ch); self.exact = {}; self.sol = {}; self.pols = {}
        self.C0 = ch.C0(); self._pc, self._pm = {}, {}
        _s.setrecursionlimit(max(20000, _s.getrecursionlimit()))
        def solve(mode, al):
            cfg = RBDPConfig(T=T, B=B, t01=ch.t01, t10=ch.t10, e0=ch.e0, e1=ch.e1, prior1=ch.prior1, c=c, k=k, alpha=al,
                             n_beta=201, mode=mode, lattice_exact=True, compute_lower=False)
            s = _Solver(cfg); s._build_post(); s._run_dp(lower=False, keep=True)
            evl = _Evaluator(s); lo, _ = s.root_indices()
            cst, rc, ri = evl.pre(1, B, lo, float(s.beta1))
            return (s, evl, lo), dict(cost=cst, risk_causal=rc, risk_iid=ri)
        for nm, mode in (("P5", "iid"), ("P7", "causal")):
            self.sol[nm], self.exact[nm] = solve(mode, alpha)
        self.exact["never"] = dict(cost=self.C0, risk_causal=0.0, risk_iid=0.0)
        pl = p9.p5r_plan(ch, alpha); self.p5r = pl
        self.exact["P5r"] = dict(cost=pl["cost"], risk_causal=pl["risk_causal"], risk_iid=pl["risk_iid"], choice=pl["choice"], alpha_prime=pl["alpha_prime"], a=pl["a"])
        if pl["choice"] == "deflated" and pl["alpha_prime"] > 0:
            self.sol["P5d"] = self.sol["P5"] if pl["alpha_prime"] == alpha else solve("iid", pl["alpha_prime"])[0]

# name: (T, B, c, k, misspec)
CONFIGS = {
    "base": (32, 4, 0.01, 0.0, None),
    "B6": (32, 6, 0.01, 0.0, None),
    "B8": (32, 8, 0.01, 0.0, None),
    "T64": (64, 4, 0.01, 0.0, None),
    "T128": (128, 4, 0.01, 0.0, None),
    "k0.05": (32, 4, 0.01, 0.05, None),
    "k0.2": (32, 4, 0.01, 0.2, None),
    "mis_persist_up": (32, 4, 0.01, 0.0, "persist_up"),
    "mis_persist_down": (32, 4, 0.01, 0.0, "persist_down"),
    "mis_emit_shrink": (32, 4, 0.01, 0.0, "emit_shrink"),
    "mis_emit_widen": (32, 4, 0.01, 0.0, "emit_widen"),
}

def perturb(sp, kind):
    sp = dict(sp)
    clip = lambda v: float(min(0.98, max(0.02, v)))
    if kind == "persist_up":      # planner believes the regime is stickier than fitted
        sp["t01"], sp["t10"] = clip(sp["t01"] * 0.5), clip(sp["t10"] * 0.5)
    elif kind == "persist_down":  # planner believes the regime switches faster
        sp["t01"], sp["t10"] = clip(sp["t01"] * 2.0), clip(sp["t10"] * 2.0)
    elif kind == "emit_shrink":   # alternative's signal is weaker than fitted
        f = 0.6  # scale around 1/2 so the emissions keep straddling 1/2
        sp["e0"], sp["e1"] = clip(0.5 + f * (sp["e0"] - 0.5)), clip(0.5 + f * (sp["e1"] - 0.5))
    elif kind == "emit_widen":    # alternative's signal is stronger than fitted
        f = 1.4
        sp["e0"], sp["e1"] = clip(0.5 + f * (sp["e0"] - 0.5)), clip(0.5 + f * (sp["e1"] - 0.5))
    return sp

def windows(cid, T):
    d, n = next((dd, nn) for c, dd, nn in p8.dataset_list() if c == cid)
    _, re_ = p8.split_episodes(p8.episodes_of(d, n))
    return [e[i * T:(i + 1) * T] for e in re_ for i in range(len(e) // T)]

def one(task):
    cfg, rec = task
    T, B, c, k, mis = CONFIGS[cfg]
    t0 = time.perf_counter()
    cid = rec["id"]
    out = {"cfg": cfg, "id": cid, "T": T, "B": B, "c": c, "k": k, "mis": mis}
    W = windows(cid, T)
    out["n_windows"] = len(W)
    sp = rec["solver_params"]
    if mis:
        sp = perturb(sp, mis)
    S = SetupLite(sp, T, B, p9.ALPHA, c, k)
    out["exact"] = S.exact
    out["C0"] = S.C0
    cost = {nm: [] for nm in POLS}; prom = {nm: [] for nm in POLS}
    for w, xs in enumerate(W):
        for nm in POLS:
            ct, pr = S.run(nm, xs, p9.SEED + w)
            cost[nm].append(ct); prom[nm].append(bool(pr))
    out.update(cost=cost, promoted=prom, wall_s=time.perf_counter() - t0, status="replayed" if W else "no window")
    return out

def census():
    return p9.census_ok()

def run(cfg, workers, subset=None):
    recs = census()
    ids = {json.loads(l)["id"] for l in open(p9.REPLAY)}
    recs = [r for r in recs if r["id"] in ids]
    if subset:
        rng = np.random.default_rng(20261125)
        recs = [recs[i] for i in sorted(rng.permutation(len(recs))[:subset])]
    done = {(json.loads(l)["cfg"], json.loads(l)["id"]) for l in open(OUT)} if OUT.exists() else set()
    todo = [(cfg, r) for r in recs if (cfg, r["id"]) not in done]
    # longest-first
    todo.sort(key=lambda t: -t[1]["n_fit_rounds"])
    print(cfg, "todo", len(todo), flush=True)
    with mp.Pool(workers) as pool, open(OUT, "a") as fh:
        for res in pool.imap_unordered(one, todo):
            fh.write(json.dumps(res) + "\n"); fh.flush()
            print(time.strftime("%H:%M:%S"), cfg, res["id"], res["status"], res["n_windows"], "%.0fs" % res["wall_s"], flush=True)

def timeit(cfg):
    recs = census()
    r = sorted(recs, key=lambda r: r["n_fit_rounds"])[len(recs) // 2]
    res = one((cfg, r)); print(cfg, r["id"], res["n_windows"], "%.1fs" % res["wall_s"])

if __name__ == "__main__":
    a = sys.argv
    w = int(next((x.split("=")[1] for x in a if x.startswith("--workers=")), 8))
    sub = next((int(x.split("=")[1]) for x in a if x.startswith("--subset=")), None)
    if a[1] == "run": run(a[2], w, sub)
    elif a[1] == "time": timeit(a[2])
