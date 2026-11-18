"""Mean±std over seeds, paired t-test across categories vs baseline run dir.
usage: python scripts/aggregate.py runs/repvit [runs/resnet18]"""
import sys, glob, json, numpy as np
from scipy import stats
def load(d):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/seed*.json"))]
    cats = sorted(runs[0])
    return {k: np.array([[r[c][k] for c in cats] for r in runs]) for k in ("i_auroc","p_auroc","pro")}
a = load(sys.argv[1])
for k, v in a.items():
    ps = v.mean(1); print(f"{k}: {ps.mean():.2f} ± {ps.std(ddof=1):.2f}")
if len(sys.argv) > 2:
    b = load(sys.argv[2])
    for k in a:
        t, p = stats.ttest_rel(a[k].mean(0), b[k].mean(0))
        print(f"{k} vs baseline: diff={a[k].mean()-b[k].mean():+.2f} t={t:.2f} p={p:.4f}")
