"""
make_fig8.py - draws Fig. 8 from the per-run file written by final_combined_experiment.py

Usage:  python make_fig8.py final_results/saf_sweep_raw.csv [out.png]

Input columns (exactly what final_combined_experiment.py writes):
    condition, seed, saf_rate, accuracy      (saf_rate is a fraction 0..0.4, accuracy a fraction 0..1)

Everything is recomputed from the raw runs: mean, sample SD (ddof=1).
Upper error bars are clipped at 100 % (cosmetic only: accuracy cannot exceed 100 %).
Legend label for the QAT condition is "FAT + QAT".
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"Baseline": "Baseline", "PTQ": "PTQ", "FAT": "FAT", "QAT": "FAT + QAT"}
ORDER = ["Baseline", "PTQ", "FAT", "QAT"]

def main(path, out="fig8_saf_sweep.png"):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    need = {"condition", "seed", "saf_rate", "accuracy"}
    if not rows or not need <= set(rows[0]):
        sys.exit(f"CSV must have columns {sorted(need)}; found {list(rows[0]) if rows else 'nothing'}")
    saf = sorted({round(float(r["saf_rate"]) * 100) for r in rows})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    print("condition, SAF%, n, mean, sd")
    for cond in ORDER:
        m, lo, hi = [], [], []
        for s in saf:
            a = np.array([float(r["accuracy"]) for r in rows
                          if r["condition"] == cond and round(float(r["saf_rate"]) * 100) == s]) * 100
            if len(a) == 0:
                sys.exit(f"missing data for {cond} at {s}% SAF")
            sd = a.std(ddof=1) if len(a) > 1 else 0.0
            m.append(a.mean()); lo.append(sd); hi.append(min(sd, 100 - a.mean()))
            print(f"{cond}, {s}, {len(a)}, {a.mean():.1f}, {sd:.1f}")
        ax.errorbar(saf, m, yerr=[lo, hi], marker="o", capsize=3, label=LABEL[cond])
    ax.set_xlabel("Stuck-At-Fault (SAF) Rate (%)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Impact of SAF Defects on Classification Accuracy")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200)
    print("saved", out)

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(*sys.argv[1:3])
