"""
uffl_controller_checks.py

Numerical checks of Eq. (2):  f_opt = min(alpha*v/max(d, eps), f_max)

  A. Table VII reproduced from Eq. (2) at alpha = 8.0
  B. What the "Duty Cycle (%)" column of Table VII actually equals (f/f_max)
  C. alpha sweep (0.5, 1.5, 8.0): finite? does clamping ever occur?
  D. Edge cases: v = 0, d < eps, d = 0, NaN input
  E. Illustration of a minimum-speed floor (PROPOSED, not in the reported firmware)

Run:  python uffl_controller_checks.py
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EPS, FMAX = 0.1, 20.0

def f_opt(d, v, alpha=8.0, eps=EPS, fmax=FMAX):
    return np.minimum(alpha * np.asarray(v, float) / np.maximum(d, eps), fmax)

rows = list(csv.DictReader(open(HERE / "table_vii_embedded_validation.csv")))

print("=== A. Table VII vs Eq. (2), alpha = 8.0 ===")
print(f"{'d':>5} {'v':>5} {'f_reported':>10} {'f_eq2':>8} {'abs diff':>9}")
maxdiff = 0
for r in rows:
    d, v, f = float(r["distance_m"]), float(r["walking_speed_mps"]), float(r["feedback_frequency_hz"])
    fe = float(f_opt(d, v)); maxdiff = max(maxdiff, abs(fe - f))
    print(f"{d:>5.2f} {v:>5.2f} {f:>10.2f} {fe:>8.3f} {abs(fe-f):>9.3f}")
print(f"max |difference| = {maxdiff:.3f} Hz\n")

print("=== B. What is the 'Duty Cycle (%)' column? ===")
print(f"{'f (Hz)':>7} {'reported %':>10} {'f/fmax*100':>11} {'PWM/255*100':>12}")
for r in rows:
    f = float(r["feedback_frequency_hz"]); rep = float(r["reported_duty_cycle_pct"]); pwm = float(r["pwm_output"])
    print(f"{f:>7.2f} {rep:>10.1f} {f/FMAX*100:>11.2f} {pwm/255*100:>12.2f}")
print("=> reported column tracks f/f_max (normalised feedback frequency), NOT MCU awake time.\n")

print("=== C. alpha sweep: v in [0,1.6] m/s, d in [0.05, 5] m ===")
dd, vv = np.meshgrid(np.linspace(0.05, 5, 500), np.linspace(0, 1.6, 200))
for a in (0.5, 1.5, 8.0):
    unclamped = a * vv / np.maximum(dd, EPS)
    f = np.minimum(unclamped, FMAX)
    print(f"alpha={a:>4}: finite={np.isfinite(f).all()}  max unclamped={unclamped.max():6.2f} Hz  "
          f"max f_opt={f.max():5.2f} Hz  clamped fraction={np.mean(unclamped >= FMAX):.3f}")
print("=> alpha = 0.5 never reaches f_max in this range; alpha = 1.5 only at v >= 1.33 m/s and d <= eps.\n")

print("=== D. Edge cases (alpha = 8.0) ===")
for d in (0.0, 0.05, 0.1, 0.5):
    print(f"d={d:>4} v=1.0 -> f_opt={float(f_opt(d,1.0)):.2f} Hz")
for d in (0.1, 0.5, 1.0):
    print(f"d={d:>4} v=0.0 -> f_opt={float(f_opt(d,0.0)):.2f} Hz   (stationary user: NO feedback)")
with np.errstate(all="ignore"):
    print(f"d=NaN v=1.0 -> f_opt={float(f_opt(np.nan,1.0))}   (invalid reading propagates as NaN)\n")

print("=== E. PROPOSED minimum-speed floor  v_eff = max(v, v_min)  (v_min = 0.3 m/s, illustrative) ===")
VMIN = 0.3
for d in (0.5, 1.0, 2.0, 4.0):
    print(f"d={d:>3} v=0.0 -> without floor {float(f_opt(d,0.0)):.2f} Hz | with floor {float(f_opt(d,max(0.0,VMIN))):.2f} Hz")
