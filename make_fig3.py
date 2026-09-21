"""Fig. 3 - Theoretical UFFL behaviour, Eq. (2) with g = 1.0 (Flat Wall / Low Obstacle).
f_cmd = min(f_max, g(c) * alpha * v_u / max(d_min, eps))
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA, EPS, F_MAX, G = 8.0, 0.1, 20.0, 1.0

d = np.linspace(0.1, 4.5, 500)
fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=200)
for v in (0.8, 1.0, 1.2, 1.4):
    f_cmd = np.minimum(F_MAX, G * ALPHA * v / np.maximum(d, EPS))
    ax.plot(d, f_cmd, lw=2, label=f"v_u = {v:.1f} m/s")
ax.axhline(F_MAX, color="gray", ls="--", lw=1, label="f_max = 20 Hz")
ax.set_xlim(0, 4.5); ax.set_ylim(0, 22)
ax.set_xlabel("Obstacle Distance, d_min (m)")
ax.set_ylabel("Adaptive Haptic Feedback Frequency, f_cmd (Hz)")
ax.set_title("Theoretical UFFL Behavior (α = 8.0)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("fig3_uffl_behavior.png")
