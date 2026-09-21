"""
classifier_controller_integration.py

Software-only error-propagation analysis of the classifier -> controller mapping
EXACTLY AS WRITTEN in Section III-A of the manuscript (Sept 21 version):

  * Flat Wall / Low Obstacle predicted  -> UFFL loop with Eq. (2) unchanged
  * Ascending / Descending Stairs       -> UFFL loop with a fixed +20 % frequency bias
  * confidence < 60 %                   -> default to the Flat Wall behaviour
  Eq. (2):  f_opt = min(alpha * v / max(d, eps), f_max)

What this script can and cannot do
  - It represents the classifier by its EMPIRICAL pooled confusion matrix at 30 % SAF
    (transcribed from Fig. 9), draws a predicted class for each true class, applies the
    policy above and reports how the commanded frequency changes.
  - It is NOT an end-to-end run of the memristive network.
  - The 60 % rejection threshold CANNOT be evaluated here: per-sample confidences are not
    in a confusion matrix. Re-run with the model's softmax outputs to evaluate it.
  - The +20 % bias and the 60 % threshold are design choices that are not validated anywhere
    in the manuscript.

Run:  python classifier_controller_integration.py
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ALPHA, EPS, FMAX = 8.0, 0.1, 20.0
CLASSES = ["Flat_Wall", "Ascending_Stairs", "Descending_Stairs", "Low_Obstacle"]
GAIN = {"Flat_Wall": 1.0, "Ascending_Stairs": 1.2, "Descending_Stairs": 1.2, "Low_Obstacle": 1.0}

def f_policy(cls, d, v):
    """frequency actually commanded when the controller believes the class is `cls`"""
    return min(GAIN[cls] * ALPHA * v / max(d, EPS), FMAX)

def load_matrices():
    m = {}
    for line in open(HERE / "confusion_matrices_30pct_saf.csv"):
        if line.startswith("#") or line.startswith("condition"):
            continue
        p = line.strip().split(",")
        m.setdefault(p[0], []).append([int(x) for x in p[2:]])
    return {k: np.array(v) for k, v in m.items()}

profile = [(float(r["distance_m"]), float(r["walking_speed_mps"]))
           for r in csv.DictReader(open(ROOT / "uffl_controller" / "table_vii_embedded_validation.csv"))]
mats = load_matrices()

print("Policy (as written in the manuscript): gain per PREDICTED class =", GAIN)
print("Frequency depends on the predicted class only through this gain; distance and speed enter via Eq. (2).\n")

rows = []
for cond in ("Baseline", "PTQ", "FAT", "FAT+QAT"):
    m = mats[cond]; P = m / m.sum(1, keepdims=True)
    print(f"=== {cond} (30% SAF, pooled over 5 runs) ===")
    print(f"  {'true class':<18}{'same gain':>10}{'gain +20%':>11}{'gain -16.7%':>13}   meaning of the last two columns")
    for i, c in enumerate(CLASSES):
        same = sum(P[i, j] for j in range(4) if GAIN[CLASSES[j]] == GAIN[c])
        up = sum(P[i, j] for j in range(4) if GAIN[CLASSES[j]] > GAIN[c])
        down = sum(P[i, j] for j in range(4) if GAIN[CLASSES[j]] < GAIN[c])
        note = ("wall/low mistaken for stairs -> frequency raised by 20%" if GAIN[c] == 1.0
                else "stairs mistaken for wall/low -> stairs bias LOST (frequency 16.7% lower)")
        print(f"  {c:<18}{same:>10.3f}{up:>11.3f}{down:>13.3f}   {note}")
        rows.append([cond, c, round(same, 4), round(up, 4), round(down, 4)])
    # cue-type accuracy (class label right) for reference
    print(f"  class-label accuracy = {np.trace(m)/m.sum():.3f}\n")

with open(HERE / "integration_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["condition", "true_class", "p_same_gain", "p_gain_up_20pct", "p_gain_down_16.7pct"])
    w.writerows(rows)

# Worked example along the Table VII approach profile: a STAIRS event predicted as wall/low (FAT+QAT, asc stairs row)
m = mats["FAT+QAT"]; P = m / m.sum(1, keepdims=True)
p_lost = sum(P[1, j] for j in (0, 3))
print("=== Worked example: ASCENDING STAIRS along the Table VII profile, FAT + QAT ===")
print(f"P(stairs bias lost) = {p_lost:.3f}")
print(f"{'d (m)':>6} {'v (m/s)':>8} {'f if stairs (Hz)':>17} {'f if predicted wall/low (Hz)':>30} {'reduction':>10}")
for d, v in profile:
    a, b = f_policy("Ascending_Stairs", d, v), f_policy("Flat_Wall", d, v)
    print(f"{d:>6.2f} {v:>8.2f} {a:>17.2f} {b:>30.2f} {100*(1-b/a):>9.1f}%")
print("\nSaved integration_results.csv")
