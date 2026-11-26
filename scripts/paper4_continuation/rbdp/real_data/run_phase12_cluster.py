"""Phase 12a: dataset-level analysis of the Phase 9 replay (no re-solving).

(1) clustered bootstrap: resample datasets with replacement, then windows within each sampled dataset;
(2) equal-weighted (per-dataset mean) results, with a dataset-level bootstrap.
Reads results/phase9_replay.jsonl, writes results/phase12a_cluster.md and .json.
"""
import json, sys
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parents[4] / "docs/paper4_continuation/rbdp/real_data/results"
SEED, NB = 20261119, 4000
SAFE = ["P1", "P2", "P6", "P3c", "P4c", "P5r"]
rows = [json.loads(l) for l in open(RES / "phase9_replay.jsonl")]
rows = [r for r in rows if r["status"] == "replayed"]
names = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P3c", "P4c", "P5r", "never"]
D = [{nm: np.array(r["cost"][nm]) for nm in names} for r in rows]
nd = len(D)
rng = np.random.default_rng(SEED)

def stat_window_weighted(idx_ds, idx_win):
    out = {nm: [] for nm in names}
    for i, w in zip(idx_ds, idx_win):
        for nm in names:
            out[nm].append(D[i][nm][w])
    return {nm: np.concatenate(v).mean() for nm, v in out.items()}

# point estimates
pw = {nm: np.concatenate([d[nm] for d in D]).mean() for nm in names}
dm = {nm: np.array([d[nm].mean() for d in D]) for nm in names}
ew = {nm: dm[nm].mean() for nm in names}

# clustered bootstrap (datasets, then windows within dataset) of the pooled window-weighted mean
cl = {nm: [] for nm in names}
for _ in range(NB):
    ids = rng.integers(0, nd, nd)
    sums = {nm: 0.0 for nm in names}; n = 0
    for i in ids:
        w = rng.integers(0, len(D[i]["P7"]), len(D[i]["P7"]))
        for nm in names:
            sums[nm] += D[i][nm][w].sum()
        n += len(w)
    for nm in names:
        cl[nm].append(sums[nm] / n)
# dataset-level bootstrap of equal-weighted means (resample datasets only)
eb = {nm: [] for nm in names}
for _ in range(NB):
    ids = rng.integers(0, nd, nd)
    for nm in names:
        eb[nm].append(dm[nm][ids].mean())

def ci(a): return np.percentile(a, [2.5, 97.5])
L = ["# Phase 12a: dataset-level uncertainty for the replay", "",
     f"{nd} datasets, {sum(len(d['P7']) for d in D)} windows. Bootstrap: {NB} resamples, seed {SEED}.",
     "Clustered: datasets resampled with replacement, then windows within each drawn dataset (window-weighted pooled mean).",
     "Equal-weighted: each dataset's mean realised cost counts once; datasets resampled.", "",
     "| policy | window-weighted mean | clustered 95% CI | equal-weighted mean | dataset 95% CI |", "|---|---|---|---|---|"]
for nm in names:
    c, e = ci(cl[nm]), ci(eb[nm])
    L.append(f"| {nm} | {pw[nm]:.4f} | [{c[0]:.4f}, {c[1]:.4f}] | {ew[nm]:.4f} | [{e[0]:.4f}, {e[1]:.4f}] |")

L += ["", "## P7 minus X (negative = P7 cheaper)", "",
      "| baseline X | window-weighted diff | clustered 95% CI | equal-weighted diff | dataset 95% CI | datasets P7 cheaper / tie / dearer | sign-test p |", "|---|---|---|---|---|---|---|"]
from math import comb
res = {}
for nm in SAFE + ["P5", "P3", "P4", "never"]:
    dpw = pw["P7"] - pw[nm]; dew = dm["P7"] - dm[nm]
    cb = np.array(cl["P7"]) - np.array(cl[nm])  # note: same resamples, paired
    eb_ = np.array(eb["P7"]) - np.array(eb[nm])
    ch, ee = ci(cb), ci(eb_)
    less = int((dew < -1e-9).sum()); tie = int((np.abs(dew) <= 1e-9).sum()); more = int((dew > 1e-9).sum())
    n = less + more
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(less, more) + 1)) / 2 ** n) if n else 1.0
    L.append(f"| {nm} | {dpw:+.4f} | [{ch[0]:+.4f}, {ch[1]:+.4f}] | {dew.mean():+.4f} | [{ee[0]:+.4f}, {ee[1]:+.4f}] | {less} / {tie} / {more} | {p:.3g} |")
    res[nm] = dict(w=dpw, e=float(dew.mean()), cl=ch.tolist(), eq=ee.tolist(), less=less, tie=tie, more=more)
# best safe per dataset
bs = np.array([min(dm[nm][i] for nm in SAFE) for i in range(nd)])
d = dm["P7"] - bs
eb_best = [ (dm["P7"][ids] - bs[ids]).mean() for ids in (rng.integers(0, nd, nd) for _ in range(NB))]
L += ["", f"Versus the per-dataset best safe baseline (P1,P2,P6,P3c,P4c,P5r): mean diff {d.mean():+.5f}, dataset 95% CI [{ci(eb_best)[0]:+.5f}, {ci(eb_best)[1]:+.5f}]; "
      f"P7 below best safe on {(d < -1e-9).sum()}, tie {(np.abs(d) <= 1e-9).sum()}, above {(d > 1e-9).sum()} of {nd} datasets."]
# VOI share, equal weighted
voi = lambda nm: (dm["never"] - dm[nm])
L += ["", "## Value-of-information share relative to P7 (dataset-level, equal-weighted: mean VOI_X / mean VOI_P7)", "", "| policy | share | dataset 95% CI |", "|---|---|---|"]
for nm in SAFE + ["P5"]:
    pt = voi(nm).mean() / voi("P7").mean()
    bb = []
    for _ in range(NB):
        ids = rng.integers(0, nd, nd); bb.append(voi(nm)[ids].mean() / voi("P7")[ids].mean())
    L.append(f"| {nm} | {pt:.3f} | [{ci(bb)[0]:.3f}, {ci(bb)[1]:.3f}] |")
open(RES / "phase12a_cluster.md", "w").write("\n".join(L) + "\n")
json.dump(res, open(RES / "phase12a_cluster.json", "w"), indent=1)
print("\n".join(L))
