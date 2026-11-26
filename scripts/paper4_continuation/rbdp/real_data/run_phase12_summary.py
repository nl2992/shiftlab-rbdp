"""Phase 12b summary: per configuration, P7 vs P5r / P5 / never on the held-out replay, with dataset-level uncertainty."""
import json
from pathlib import Path
import numpy as np
RES = Path(__file__).resolve().parents[4] / "docs/paper4_continuation/rbdp/real_data/results"
ALPHA, NB, SEED = 0.1, 4000, 20261119
rows = [json.loads(l) for l in open(RES / "phase12b_robust.jsonl")]
order = ["base", "B6", "B8", "T64", "T128", "k0.05", "k0.2", "mis_persist_up", "mis_persist_down", "mis_emit_shrink", "mis_emit_widen"]
L = ["# Phase 12b: replay robustness (P7 vs the strongest safe baseline P5r)", "",
     "alpha = 0.1, c = 0.01; one factor changed per row. Dataset = unit of analysis (equal-weighted mean over datasets with at least one window); "
     "CI = bootstrap over datasets (4000, seed 20261119). diff = P7 - P5r (negative = P7 cheaper). "
     "'exact risk' = max over datasets of exact worst-case adaptive risk / alpha under the (possibly misspecified) planning model.", "",
     "| config | T | B | k | planning-model change | datasets | windows | P7 | P5r | P5 (iid) | never | P7 - P5r [95% CI] | P7 < P5r / tie / > | P7 - P5 [95% CI] | VOI share P5r | max exact risk/alpha P7 | P5 | P5r |",
     "|" + "---|" * 18]
rng = np.random.default_rng(SEED)
for cfg in order:
    R = [r for r in rows if r["cfg"] == cfg and r["status"] == "replayed"]
    if not R: continue
    m = {p: np.array([np.mean(r["cost"][p]) for r in R]) for p in ("P5", "P7", "P5r", "never")}
    d = m["P7"] - m["P5r"]; d5 = m["P7"] - m["P5"]
    ids = [rng.integers(0, len(R), len(R)) for _ in range(NB)]
    ci = lambda x: np.percentile([x[i].mean() for i in ids], [2.5, 97.5])
    c1, c5 = ci(d), ci(d5)
    tol = 1e-9
    voi5r = (m["never"] - m["P5r"]).mean() / (m["never"] - m["P7"]).mean()
    rk = lambda p: max(r["exact"][p]["risk_causal"] for r in R) / ALPHA
    r0 = R[0]
    L.append(f"| {cfg} | {r0['T']} | {r0['B']} | {r0['k']} | {r0['mis'] or '-'} | {len(R)} | {sum(r['n_windows'] for r in R)} | "
             f"{m['P7'].mean():.3f} | {m['P5r'].mean():.3f} | {m['P5'].mean():.3f} | {m['never'].mean():.3f} | "
             f"{d.mean():+.4f} [{c1[0]:+.4f}, {c1[1]:+.4f}] | {(d < -tol).sum()} / {(abs(d) <= tol).sum()} / {(d > tol).sum()} | "
             f"{d5.mean():+.4f} [{c5[0]:+.4f}, {c5[1]:+.4f}] | {voi5r:.3f} | {rk('P7'):.3f} | {rk('P5'):.2f} | {rk('P5r'):.3f} |")
# in-model values (exact, no real labels) per config
L += ["", "## In-model exact comparison (fitted/planning model, no real labels): mean over datasets", "",
      "| config | datasets | mean exact P7 | mean exact P5r | mean (P5r - P7) | datasets with P5r - P7 > 1e-9 | max (P5r - P7) | mean C0 - P7 (VOI) |", "|---|---|---|---|---|---|---|---|"]
for cfg in order:
    R = [r for r in rows if r["cfg"] == cfg]
    if not R: continue
    g = np.array([r["exact"]["P5r"]["cost"] - r["exact"]["P7"]["cost"] for r in R])
    L.append(f"| {cfg} | {len(R)} | {np.mean([r['exact']['P7']['cost'] for r in R]):.4f} | {np.mean([r['exact']['P5r']['cost'] for r in R]):.4f} | "
             f"{g.mean():.5f} | {(g > 1e-9).sum()} | {g.max():.4f} | {np.mean([r['C0'] - r['exact']['P7']['cost'] for r in R]):.4f} |")
open(RES / "phase12b_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
