"""
make_table8.py - Table VIII and the Fig. 7 caption numbers from the distance log.

Usage:  python make_table8.py distance_log.csv
Input columns: set_cm, measured_cm      (one row per test point, all 9 points)
Prints the table (with absolute and relative error) and the caption maxima.
"""
import sys, csv
rows = [(float(r["set_cm"]), float(r["measured_cm"]))
        for r in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))]
print("Set (cm) | Measured (cm) | Abs err (cm) | Rel err (%)")
worst_a = worst_r = None
for s, m in rows:
    a, r = abs(m - s), abs(m - s) / s * 100
    print(f"{s:g} | {m:.2f} | {a:.2f} | {r:.2f}")
    worst_a = max(worst_a or (a, s), (a, s)); worst_r = max(worst_r or (r, s), (r, s))
print(f"\nn = {len(rows)} points")
print(f"max absolute error: {worst_a[0]:.2f} cm at {worst_a[1]:g} cm")
print(f"max relative error: {worst_r[0]:.2f} % at {worst_r[1]:g} cm")
