"""
per_class_metrics.py

Per-class precision / recall / F1 and macro-F1 from the POOLED confusion matrices
of Fig. 9 (5 runs summed, 30% SAF).

CAVEATS
* Counts are pooled over five runs that share the same task; they are not five
  independent test sets, so no confidence intervals are attached.
* The matrices in confusion_matrices_30pct_saf.csv were transcribed from Fig. 9.
  Re-run this script on the raw per-run matrices in the experiment repository.
Run:  python per_class_metrics.py
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CLASSES = ["Flat_Wall", "Ascending_Stairs", "Descending_Stairs", "Low_Obstacle"]

mats = {}
for line in open(HERE / "confusion_matrices_30pct_saf.csv"):
    if line.startswith("#") or line.startswith("condition"):
        continue
    p = line.strip().split(",")
    mats.setdefault(p[0], []).append([int(x) for x in p[2:]])
mats = {k: np.array(v) for k, v in mats.items()}

out_rows = []
for cond, m in mats.items():
    tp = np.diag(m).astype(float)
    support = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = tp / m.sum(0)
        rec = tp / support
        f1 = 2 * prec * rec / (prec + rec)
    print(f"\n{cond}: total={m.sum()}  accuracy={tp.sum()/m.sum():.4f}  macro-F1={np.nanmean(f1):.3f}")
    print(f"  {'class':<18}{'support':>8}{'prec':>8}{'recall':>8}{'F1':>8}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:<18}{support[i]:>8}{prec[i]:>8.3f}{rec[i]:>8.3f}{f1[i]:>8.3f}")
        out_rows.append([cond, c, support[i], round(prec[i], 4), round(rec[i], 4), round(f1[i], 4)])
    out_rows.append([cond, "MACRO_AVG", int(m.sum()), round(np.nanmean(prec), 4), round(np.nanmean(rec), 4), round(np.nanmean(f1), 4)])

with open(HERE / "per_class_metrics.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["condition", "class", "support", "precision", "recall", "f1"]); w.writerows(out_rows)
print("\nSaved per_class_metrics.csv")
