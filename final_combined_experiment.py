"""
final_combined_experiment.py

Single controlled experiment for Table X + Figure 8.

Design:
- Same fixed dataset/split for every condition and seed.
- Seed is set before model construction.
- One trained reference model is created per condition/seed.
- PTQ/QAT quantization is applied once to a reference copy.
- For each SAF level, a DEEP COPY of the reference model is passed to
  MemTorch, so one SAF evaluation cannot modify another.
- The 30% SAF accuracy used for Table X is taken directly from the same
  stored sweep rows that generate Figure 8.
- Per-run results, summary statistics, confusion matrices at 30% SAF,
  SAF sweep summary, Figure 8, and environment are all written together.

This script intentionally refuses to use a non-MemTorch fallback.
"""

import csv
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from dataset import generate_dataset, CONFIG as DATA_CONFIG, CLASS_NAMES
from model import ObstacleMLP
from saf import (
    SAF_LEVELS,
    build_memtorch_eval_model,
    MEMTORCH_AVAILABLE,
    forward_condition,
    quantize_model_weights,
    CRITICAL_SAF_LEVEL,
)

TRAIN_CONFIG = {
    "epochs": 100,
    "lr": 1e-3,
    "batch_size": 32,
    "quant_bits": 4,
    "train_saf_rate": CRITICAL_SAF_LEVEL,
}

SEEDS = [0, 1, 2, 3, 4]
DATASET_MASTER_SEED = 42
REGENERATE_DATA_PER_SEED = False
CONDITIONS = ["Baseline", "PTQ", "FAT", "QAT"]
OUT_DIR = "final_results"


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, X_train, y_train, condition: str, seed: int):
    """Train one model using the corrected differentiable FAT/QAT path."""
    set_all_seeds(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])
    loss_fn = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    n = len(y_train_t)

    model.train()

    for epoch in range(TRAIN_CONFIG["epochs"]):
        perm = torch.randperm(n)

        for batch_id, i in enumerate(range(0, n, TRAIN_CONFIG["batch_size"])):
            idx = perm[i:i + TRAIN_CONFIG["batch_size"]]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad(set_to_none=True)

            fault_seed = seed * 1_000_000 + epoch * 10_000 + batch_id

            out = forward_condition(
                model,
                xb,
                condition=condition,
                saf_rate=TRAIN_CONFIG["train_saf_rate"],
                quant_bits=TRAIN_CONFIG["quant_bits"],
                fault_seed=fault_seed,
            )

            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

    return model


def evaluate_memtorch(reference_model, X_test, y_test, seed, saf_rate):
    """
    Evaluate an untouched reference model through a fresh MemTorch copy.

    The MemTorch evaluation helper performs clean tuning first, then
    DeviceFault injection, using the supplied seed.
    """
    # Reset all RNG sources immediately before each independent evaluation.
    set_all_seeds(seed)

    model_for_eval = copy.deepcopy(reference_model).cpu()
    eval_model = build_memtorch_eval_model(
        model_for_eval,
        saf_rate,
        seed,
    )

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    eval_model.eval()

    with torch.no_grad():
        logits = eval_model(X_test_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    acc = float((preds == y_test).mean())
    cm = confusion_matrix(
        y_test,
        preds,
        labels=list(range(DATA_CONFIG["num_classes"])),
    )
    return acc, cm


def log_environment():
    os.makedirs(OUT_DIR, exist_ok=True)

    lines = [
        f"python: {sys.version}",
        f"torch: {torch.__version__}",
        f"numpy: {np.__version__}",
        f"dataset_master_seed: {DATASET_MASTER_SEED}",
        f"seeds: {SEEDS}",
        f"classes: {CLASS_NAMES}",
        f"train_config: {TRAIN_CONFIG}",
        f"critical_saf_level: {CRITICAL_SAF_LEVEL}",
        f"memtorch_available: {MEMTORCH_AVAILABLE}",
    ]

    try:
        import memtorch
        lines.append(f"memtorch: {getattr(memtorch, '__version__', 'unknown')}")
    except ImportError:
        lines.append("memtorch: NOT INSTALLED")

    try:
        import sklearn
        lines.append(f"scikit-learn: {sklearn.__version__}")
    except ImportError:
        pass

    with open(
        os.path.join(OUT_DIR, "environment.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


def run_all():
    if not MEMTORCH_AVAILABLE:
        raise RuntimeError(
            "MemTorch is unavailable. The final experiment refuses "
            "to generate non-MemTorch numbers."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    log_environment()

    fixed_data = generate_dataset(master_seed=DATASET_MASTER_SEED)
    X_train, y_train = fixed_data["train"]
    X_test, y_test = fixed_data["test"]

    print(
        f"Dataset: {fixed_data['total_samples']} samples; "
        f"train={len(y_train)}, test={len(y_test)}"
    )

    raw_rows = []
    cm_by_condition = {
        condition: np.zeros(
            (DATA_CONFIG["num_classes"], DATA_CONFIG["num_classes"]),
            dtype=int,
        )
        for condition in CONDITIONS
    }

    # Store every SAF evaluation in memory; Table X at 30% is taken from here.
    sweep_rows = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            # IMPORTANT: seed BEFORE model construction.
            set_all_seeds(seed)

            print(
                f"\nTraining {condition}, seed={seed}..."
            )

            model = ObstacleMLP()
            model = train_model(
                model,
                X_train,
                y_train,
                condition,
                seed,
            )

            reference_model = copy.deepcopy(model)

            # Match the same PTQ/QAT preparation used for Table X.
            if condition in ("PTQ", "QAT"):
                quantize_model_weights(
                    reference_model,
                    TRAIN_CONFIG["quant_bits"],
                )

            for saf_rate in SAF_LEVELS:
                acc, cm = evaluate_memtorch(
                    reference_model,
                    X_test,
                    y_test,
                    seed + 100_000,
                    saf_rate,
                )

                sweep_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "saf_rate": saf_rate,
                        "accuracy": acc,
                    }
                )

                print(
                    f"  SAF={saf_rate * 100:.0f}% "
                    f"accuracy={acc * 100:.1f}%"
                )

                if saf_rate == CRITICAL_SAF_LEVEL:
                    raw_rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "accuracy": acc,
                        }
                    )
                    cm_by_condition[condition] += cm

    # ----------------------------
    # Save raw Table X observations
    # ----------------------------
    with open(
        os.path.join(OUT_DIR, "results.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "seed", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    # ----------------------------
    # Table X summary, sourced from raw_rows
    # ----------------------------
    with open(
        os.path.join(OUT_DIR, "summary.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition",
            "n_runs",
            "mean_accuracy_pct",
            "sample_sd_accuracy_pct",
        ])

        for condition in CONDITIONS:
            accs = np.array([
                r["accuracy"]
                for r in raw_rows
                if r["condition"] == condition
            ]) * 100

            sd = accs.std(ddof=1) if len(accs) > 1 else 0.0

            writer.writerow([
                condition,
                len(accs),
                f"{accs.mean():.1f}",
                f"{sd:.1f}",
            ])

            print(
                f"TABLE X ROW -> {condition}: "
                f"{accs.mean():.1f}% +/- {sd:.1f}% "
                f"(N={len(accs)}, seeds={SEEDS}, sample SD)"
            )

    # ----------------------------
    # Save all SAF sweep observations
    # ----------------------------
    with open(
        os.path.join(OUT_DIR, "saf_sweep_raw.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "seed", "saf_rate", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(sweep_rows)

    # ----------------------------
    # SAF sweep summary
    # ----------------------------
    with open(
        os.path.join(OUT_DIR, "saf_sweep_summary.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition",
            "saf_rate_pct",
            "n_runs",
            "mean_accuracy_pct",
            "sample_sd_accuracy_pct",
        ])

        for condition in CONDITIONS:
            for saf_rate in SAF_LEVELS:
                accs = np.array([
                    r["accuracy"]
                    for r in sweep_rows
                    if r["condition"] == condition
                    and r["saf_rate"] == saf_rate
                ]) * 100

                sd = accs.std(ddof=1) if len(accs) > 1 else 0.0

                writer.writerow([
                    condition,
                    f"{saf_rate * 100:.0f}",
                    len(accs),
                    f"{accs.mean():.1f}",
                    f"{sd:.1f}",
                ])

    # ----------------------------
    # Confusion matrices at 30% SAF
    # ----------------------------
    import matplotlib.pyplot as plt

    for condition, cm in cm_by_condition.items():
        np.save(
            os.path.join(
                OUT_DIR,
                f"confusion_matrix_{condition}.npy",
            ),
            cm,
        )

        fig, ax = plt.subplots()
        im = ax.imshow(cm)
        ax.set_title(
            f"Confusion Matrix - {condition} "
            f"(summed over {len(SEEDS)} runs, 30% SAF)"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                )

        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                OUT_DIR,
                f"confusion_matrix_{condition}.png",
            ),
            dpi=200,
        )
        plt.close(fig)

    # ----------------------------
    # Figure 8 from the exact same sweep rows
    # ----------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    for condition in CONDITIONS:
        means = []
        sds = []

        for saf_rate in SAF_LEVELS:
            accs = np.array([
                r["accuracy"]
                for r in sweep_rows
                if r["condition"] == condition
                and r["saf_rate"] == saf_rate
            ]) * 100

            means.append(accs.mean())
            sds.append(accs.std(ddof=1) if len(accs) > 1 else 0.0)

        x = [s * 100 for s in SAF_LEVELS]

        ax.errorbar(
            x,
            means,
            yerr=sds,
            marker="o",
            label=condition,
            capsize=3,
        )

    ax.set_xlabel("Stuck-At-Fault (SAF) Rate (%)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Impact of SAF Defects on Classification Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            OUT_DIR,
            "figure8_saf_sweep.png",
        ),
        dpi=200,
    )
    plt.close(fig)

    # ----------------------------
    # Cross-check: Table X 30% rows must equal sweep 30% rows
    # ----------------------------
    print("\n30% SAF consistency check:")

    for condition in CONDITIONS:
        table_values = sorted(
            r["accuracy"]
            for r in raw_rows
            if r["condition"] == condition
        )
        sweep_values = sorted(
            r["accuracy"]
            for r in sweep_rows
            if r["condition"] == condition
            and r["saf_rate"] == CRITICAL_SAF_LEVEL
        )

        same = np.allclose(table_values, sweep_values, rtol=0, atol=1e-12)

        print(
            f"  {condition}: "
            f"{'PASS' if same else 'FAIL'}"
        )

        if not same:
            raise RuntimeError(
                f"30% SAF mismatch between Table X and Figure 8 "
                f"rows for {condition}."
            )

    print(f"\nFigure 8 saved to {OUT_DIR}/figure8_saf_sweep.png")
    print(f"All final results saved under ./{OUT_DIR}/")


if __name__ == "__main__":
    run_all()
