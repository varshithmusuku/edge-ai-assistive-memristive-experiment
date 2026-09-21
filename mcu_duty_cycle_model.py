"""
mcu_duty_cycle_model.py

Analytical MCU duty-cycle / energy model (Section III-D/E of the manuscript).

  D          = r_event * t_awake                        (fraction of time awake, D << 1)
  I_avg      = D * I_active + (1 - D) * I_sleep  (+ wake-up term)
  E_MCU / T  = V_DD * (I_active*t_active + I_sleep*t_sleep) + N_wake * E_wake

PART 1  Reproduces Tables II and III arithmetic and shows that 4.25 mW = 0.85 mA x 5 V.
PART 2  Shows the reported "Duty Cycle (%)" of Table VII is NOT MCU awake time:
        controller execution alone (17-44 us) occupies <0.05% of each second.
PART 3  Illustrates how sleep current and per-event awake time would change the
        92.5% figure.  ALL NUMBERS IN PART 3 ARE SWEEP VALUES, NOT DEVICE DATA.
Run:  python mcu_duty_cycle_model.py
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
I_ACTIVE_MA = 11.4     # value used in the manuscript (source still to be stated by the authors)
VDD = 5.0              # V, as used to obtain 4.25 mW

print("=== PART 1: arithmetic of Tables II and III (I_sleep = 0, no wake-up term, as in the paper) ===")
print(f"{'duty':>6} {'I_avg (mA)':>11} {'reduction':>10}")
for D in (1.0, 0.30, 0.15, 0.075, 0.05, 0.225):
    I = D * I_ACTIVE_MA
    print(f"{D*100:>5.1f}% {I:>11.3f} {100*(1-D):>9.1f}%")
print(f"4.25 mW check: 0.85 mA x {VDD} V = {0.85*VDD:.2f} mW  (i.e. an AVERAGE-power scenario, not 'active power')\n")

print("=== PART 2: controller-only awake fraction from Table VII ===")
rows = list(csv.DictReader(open(ROOT / "uffl_controller" / "table_vii_embedded_validation.csv")))
print(f"{'f (Hz)':>7} {'t_exec (us)':>12} {'reported duty %':>16} {'f*t_exec %':>11}   evaluated at fixed r_eval: 10 Hz | 20 Hz | 50 Hz")
out = []
for r in rows:
    f = float(r["feedback_frequency_hz"]); t = float(r["controller_time_us"]) * 1e-6
    rep = float(r["reported_duty_cycle_pct"])
    vals = [100 * rate * t for rate in (10, 20, 50)]
    print(f"{f:>7.2f} {t*1e6:>12.0f} {rep:>16.1f} {100*f*t:>11.4f}   {vals[0]:.3f}% | {vals[1]:.3f}% | {vals[2]:.3f}%")
    out.append([f, t*1e6, rep, 100*f*t, *vals])
lo = min(o[3] for o in out); hi = max(o[3] for o in out)
print(f"=> controller arithmetic alone: {lo:.4f}% - {hi:.4f}% of each second (one evaluation per pulse); "
      f"even at 50 evaluations/s it is <= {max(o[6] for o in out):.2f}%.")
print("   The reported 7.9-55.5% therefore cannot be MCU awake time computed from these execution times.\n")

print("=== PART 2b: awake time if the MCU busy-waits for an HC-SR04 echo (illustrative) ===")
t_echo_max = 2 * 4.0 / 343.0
print(f"max echo wait for 4 m = {t_echo_max*1e3:.1f} ms")
for rate in (5, 10, 20):
    print(f"  {rate:>2} measurements/s x {t_echo_max*1e3:.1f} ms = {100*rate*t_echo_max:.1f}% awake (if blocking pulseIn-style wait is used)")
print("=> awake time depends on sensing/wake-up design, which the reported runs did not measure.\n")

print("=== PART 3: sensitivity of the 92.5% figure (SWEEP VALUES, not datasheet values) ===")
D = 0.075
print(f"{'I_sleep (mA)':>13} {'I_avg (mA)':>11} {'reduction vs 11.4 mA':>21}")
for Isl in (0.0, 0.01, 0.1, 1.0):
    I = D * I_ACTIVE_MA + (1 - D) * Isl
    print(f"{Isl:>13.3f} {I:>11.3f} {100*(1-I/I_ACTIVE_MA):>20.1f}%")
print()
print(f"{'t_awake/event (ms)':>19} {'events/s':>9} {'D = r*t':>9}")
for t_ms in (1, 5, 20):
    for r in (2, 10):
        print(f"{t_ms:>19} {r:>9} {100*r*t_ms*1e-3:>8.1f}%")

with open(HERE / "mcu_duty_cycle_model_output.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["f_hz", "t_exec_us", "reported_duty_pct", "controller_only_pct", "pct_at_10hz", "pct_at_20hz", "pct_at_50hz"])
    w.writerows(out)
print("\nSaved mcu_duty_cycle_model_output.csv")
