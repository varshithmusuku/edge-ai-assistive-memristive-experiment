"""
make_fig5.py  -  Fig. 5 of the manuscript, regenerated from Table VII data.
Panel (b) is relabelled: it plots the NORMALISED FEEDBACK FREQUENCY f/f_max (%),
not MCU awake time.
Run:  python make_fig5.py
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FMAX = 20.0
rows = list(csv.DictReader(open(HERE / "table_vii_embedded_validation.csv")))
d = [float(r["distance_m"]) for r in rows]
f = [float(r["feedback_frequency_hz"]) for r in rows]
pwm = [float(r["pwm_output"]) for r in rows]
norm = [float(r["reported_duty_cycle_pct"]) for r in rows]   # equals f/f_max*100

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(16.7, 4.9), dpi=100)
specs = [
    (f,   "s", "(a) Frequency Scaling",                        "Feedback Frequency (Hz)"),
    (norm, "o", "(b) Normalized Feedback Frequency",           "f / f$_{max}$ (%)"),
    (pwm, "^", "(c) Adaptive Actuation",                       "PWM Output (0-255)"),
]
for a, (y, m, title, yl) in zip(ax, specs):
    a.plot(d, y, color="black", marker=m, markersize=8, linewidth=2)
    a.set_title(title, fontsize=12)
    a.set_xlabel("Distance to Obstacle (m)", fontweight="bold", fontsize=11)
    a.set_ylabel(yl, fontweight="bold", fontsize=11)
    a.invert_xaxis()
    a.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
out = HERE / "fig5_embedded_validation_metrics.png"
fig.savefig(out, dpi=100)
print("Saved", out.name)
