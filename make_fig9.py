"""Fig. 9 - confusion matrices summed over 5 runs at 30% SAF.
Counts transcribed from the manuscript figure; row sums are checked against the
per-class test counts (76+66+70+88 = 300 per run -> x5 = 380/330/350/440)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

classes = ["Flat Wall", "Ascending\nStairs", "Descending\nStairs", "Low Obstacle"]
CM = {
    "Baseline":  [[116,158, 69, 37],[ 42,222,  0, 66],[140, 70,140,  0],[  0,176,  0,264]],
    "PTQ":       [[156,157, 67,  0],[  6,205,  0,119],[140, 70,140,  0],[  0,176,  0,264]],
    "FAT":       [[ 63, 86,155, 76],[  0,212,  0,118],[  0,  0,350,  0],[  0,  0,  0,440]],
    "FAT + QAT": [[ 75,139,149, 17],[  0,200,  0,130],[  0,  0,350,  0],[  0,  0,  0,440]],
}
ROW_TOTALS = [380, 330, 350, 440]
for k, m in CM.items():
    assert np.array(m).sum(axis=1).tolist() == ROW_TOTALS, k

fig, axes = plt.subplots(2, 2, figsize=(10.24, 9.17), dpi=200)
for ax, (name, m) in zip(axes.ravel(), CM.items()):
    m = np.array(m)
    im = ax.imshow(m, cmap="viridis", aspect="auto")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"Confusion Matrix - {name} (summed over 5 runs, 30% SAF)", fontsize=10)
    for i in range(4):
        for j in range(4):
            rgba = im.cmap(im.norm(m[i, j]))
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=9,
                    fontweight="bold", color="black" if lum > 0.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig("fig9_confusion_matrices.png")
print("ok")
