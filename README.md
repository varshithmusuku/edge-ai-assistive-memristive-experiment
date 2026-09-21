# edge-ai-assistive-memristive-experiment

Code, data and logs accompanying the manuscript on an edge-AI assistive obstacle-feedback
system (User-First Feedback Loop, UFFL) with a memristive-crossbar neural classifier.

**Scope.** Everything here is a *simulation study*: the classifier is evaluated with the
MemTorch/VTEAM crossbar simulator on a **synthetic** dataset, and the controller timing comes
from a simulator. No physical hardware measurements are included, and the energy figures are
scenario-based estimates, not measurements.

## 1. Environment

MemTorch's CPU wheel supports Python 3.7-3.9 only, so use **Python 3.9**.

```bash
python3.9 -m venv memtorch-paper39
# Windows:  memtorch-paper39\Scripts\activate      Linux/macOS:  source memtorch-paper39/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Pinned versions: `torch==1.10.0+cpu`, `torchvision==0.11.1+cpu`, `memtorch-cpu==1.1.6`
(plus numpy, scikit-learn, matplotlib). The exact versions used for the reported run are
recorded in `environment.txt`.

## 2. Reproduce the classifier results (Table X, Fig. 8, Fig. 9)

```bash
python smoke_test.py                 # 1-seed pipeline check; every line must end in "-> OK"
python final_combined_experiment.py  # the single experiment behind Table X, Fig. 8 and Fig. 9
python make_fig8.py final_results/saf_sweep_raw.csv figure8_saf_sweep.png
```

`final_combined_experiment.py` is the authoritative script. For each of 4 conditions
(Baseline, PTQ, FAT, QAT) and 5 seeds (0-4) it trains one 64-32-32-4 MLP and evaluates a
fresh copy of it through MemTorch at 9 SAF levels (0-40 %). The 30 % values used for
Table X are taken from the *same* stored rows as Fig. 8, and the script asserts this.

* **Data:** synthetic, 4 classes x 500 samples = 2000, split 70/15/15 (1400/300/300),
  generated from master seed 42 and held fixed across all conditions and seeds. The
  300-sample test split contains 76 / 66 / 70 / 88 samples of
  Flat Wall / Ascending Stairs / Descending Stairs / Low Obstacle
  (x 5 runs = 380 / 330 / 350 / 440 rows per confusion matrix).
* **Training:** 100 epochs, Adam, lr 1e-3, batch 32. FAT/QAT train with stuck-at faults at
  the 30 % level using a straight-through estimator; QAT additionally uses 4-bit
  fake quantisation. PTQ quantises the trained weights to 4 bits.
* **Evaluation:** MemTorch/VTEAM crossbar, clean calibration (`tune_()`) first, then
  `DeviceFaults` injection with equal stuck-low / stuck-high proportions (SAF rate / 2 each).
* **Statistics:** mean and *sample* SD (ddof = 1) over N = 5 runs.
* **Figure 8 error bars:** upper bars are clipped at 100 % (accuracy cannot exceed 100 %);
  `make_fig8.py` recomputes all means/SDs from `saf_sweep_raw.csv`.

## 3. Repository contents

| Item | Purpose |
|---|---|
| `dataset.py`, `model.py`, `saf.py` | synthetic dataset, MLP, SAF / quantisation / MemTorch helpers |
| `final_combined_experiment.py` | single experiment producing Table X, Fig. 8 sweep and confusion matrices |
| `smoke_test.py` | one-seed pipeline check |
| `make_fig8.py` | Fig. 8 from `saf_sweep_raw.csv` |
| `make_fig3.py`, `make_fig5.py`, `make_fig9.py`, `make_table8.py` | Figs. 3, 5, 9 and Table VIII |
| `uffl_alpha_derivation.py`, `uffl_controller_checks.py` | UFFL controller parameter derivation and checks |
| `mcu_duty_cycle_model.py` | MCU duty-cycle *scenario* model (assumed awake time) |
| `per_class_metrics.py`, `classifier_controller_integration.py` | per-class metrics; classifier-to-controller integration check |
| `saf_sweep_raw.csv`, `saf_sweep_summary.csv`, `results.csv`, `summary.csv` | raw and summarised results |
| `confusion_matrix_*.npy/.png`, `confusion_matrices_30pct_saf.csv` | pooled 30 % SAF confusion matrices |
| `log_*.txt`, `environment.txt`, `SHA256SUMS.txt` | run logs, software versions, file checksums |
| `experiment.py`, `saf_sweep.py` (under `legacy/`) | earlier separate runners, superseded by `final_combined_experiment.py` |

## 4. Integrity

`SHA256SUMS.txt` is provided as a file-integrity record for the released materials.

## 5. Limitations

Synthetic data with idealised geometry; simulated (not fabricated) devices; simulator-reported
timing; assumed-duty-cycle energy scenarios. See the manuscript for the bounded claims.
