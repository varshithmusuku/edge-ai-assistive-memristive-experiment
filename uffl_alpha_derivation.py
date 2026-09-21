"""
uffl_alpha_derivation.py

Purpose
-------
1. Back-compute the sensitivity coefficient alpha implied by Table VII of the
   manuscript, using  f_opt = min(alpha * v_u / max(d_min, eps), f_max)  (Eq. 2)
   ->  alpha = f * d / v   (valid because no row of Table VII is clamped).
2. Regenerate Fig. 3 using that alpha, so that Table VII and Fig. 3 agree.

Notes
-----
* Data are read from table_vii_embedded_validation.csv (same folder).
* The script FAILS (assert) if the implied alpha is not within 0.05 of 8.0,
  instead of printing a hard-coded "confirmed" message.

Run:  python uffl_alpha_derivation.py
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ALPHA, EPS, FMAX = 8.0, 0.1, 20.0

rows = []
with open(HERE / "table_vii_embedded_validation.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        rows.append((float(r["distance_m"]), float(r["walking_speed_mps"]),
                     float(r["feedback_frequency_hz"])))

print(f"{'d (m)':>6} {'v (m/s)':>8} {'f (Hz)':>7} {'clamped?':>9} {'implied alpha':>14}")
implied = []
for d, v, f in rows:
    clamped = f >= FMAX - 1e-9
    a = f * d / v
    implied.append(a)
    print(f"{d:>6.2f} {v:>8.2f} {f:>7.2f} {str(clamped):>9} {a:>14.3f}")

implied = np.array(implied)
print(f"\nimplied alpha: min={implied.min():.3f}  max={implied.max():.3f}  "
      f"mean={implied.mean():.3f}  sd={implied.std(ddof=1):.3f}")
assert np.all(np.abs(implied - ALPHA) < 0.05), "Table VII is NOT consistent with alpha = 8.0"
print(f"CHECK PASSED: all 7 rows are within 0.05 of alpha = {ALPHA}")

# ---- Fig. 3 -------------------------------------------------------------
d = np.linspace(0.1, 4.5, 400)
plt.figure(figsize=(6.5, 4.0), dpi=200)
for v in [0.8, 1.0, 1.2, 1.4]:
    f = np.minimum(ALPHA * v / np.maximum(d, EPS), FMAX)
    plt.plot(d, f, label=f"v_u = {v:.1f} m/s", linewidth=2)
plt.axhline(FMAX, color="gray", linestyle="--", linewidth=1, label=f"f_max = {int(FMAX)} Hz")
plt.xlabel("Obstacle Distance, d_min (m)")
plt.ylabel("Adaptive Haptic Feedback Frequency, f_opt (Hz)")
plt.title(f"Theoretical UFFL Behavior (\u03b1 = {ALPHA:.1f})")
plt.legend(loc="upper right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.ylim(0, 22)
plt.xlim(0, 4.5)
plt.tight_layout()
out = HERE / "fig3_theoretical_uffl_behavior.png"
plt.savefig(out, dpi=200)
print(f"Saved {out.name}")
