"""
saf_sweep.py - corrected Figure 8 SAF sweep.

The same trained model for each condition/seed is evaluated across all SAF levels.
Each SAF evaluation receives a fresh copy of the reference weights so one point
cannot modify the next point.
"""

import copy
import csv
import os

import numpy as np
import torch

from dataset import generate_dataset
from model import ObstacleMLP
from saf import (
    SAF_LEVELS,
    build_memtorch_eval_model,
    MEMTORCH_AVAILABLE,
)
from experiment import (
    train_model,
    quantize_model_weights,
    TRAIN_CONFIG,
    SEEDS,
    DATASET_MASTER_SEED,
    CONDITIONS,
)

OUT_DIR = "results"


def evaluate_at_saf_level(
    reference_model,
    X_test,
    y_test,
    seed,
    saf_rate,
):
    eval_model = build_memtorch_eval_model(
        reference_model,
        saf_rate,
        seed + 100_000,
    )

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    eval_model.eval()

    with torch.no_grad():
        preds = eval_model(X_test_t).argmax(dim=1).cpu().numpy()

    return float((preds == y_test).mean())


def run_sweep():
    if not MEMTORCH_AVAILABLE:
        raise RuntimeError(
            "MemTorch is unavailable. Refusing to generate Figure 8."
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    fixed_data = generate_dataset(master_seed=DATASET_MASTER_SEED)
    X_train, y_train = fixed_data["train"]
    X_test, y_test = fixed_data["test"]

    rows = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            # IMPORTANT: seed BEFORE constructing the model.
            torch.manual_seed(seed)
            np.random.seed(seed)

            print(
                f"Training {condition} seed={seed} for sweep..."
            )

            model = ObstacleMLP()
            model = train_model(
                model,
                X_train,
                y_train,
                condition,
                seed,
            )

            # One untouched reference copy for every SAF point.
            reference_model = copy.deepcopy(model)

            # Apply PTQ/QAT quantization once before the sweep.
            if condition in ("PTQ", "QAT"):
                quantize_model_weights(
                    reference_model,
                    TRAIN_CONFIG["quant_bits"],
                )

            for saf_rate in SAF_LEVELS:
                acc = evaluate_at_saf_level(
                    reference_model,
                    X_test,
                    y_test,
                    seed,
                    saf_rate,
                )

                rows.append(
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

    with open(
        os.path.join(OUT_DIR, "saf_sweep_raw.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "seed",
                "saf_rate",
                "accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(
        os.path.join(OUT_DIR, "saf_sweep_summary.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                "saf_rate_pct",
                "n_runs",
                "mean_accuracy_pct",
                "sample_sd_accuracy_pct",
            ]
        )

        for condition in CONDITIONS:
            for saf_rate in SAF_LEVELS:
                accs = np.array(
                    [
                        r["accuracy"]
                        for r in rows
                        if r["condition"] == condition
                        and r["saf_rate"] == saf_rate
                    ]
                ) * 100

                sd = accs.std(ddof=1) if len(accs) > 1 else 0.0

                writer.writerow(
                    [
                        condition,
                        f"{saf_rate * 100:.0f}",
                        len(accs),
                        f"{accs.mean():.1f}",
                        f"{sd:.1f}",
                    ]
                )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    for condition in CONDITIONS:
        means = []
        sds = []

        for saf_rate in SAF_LEVELS:
            accs = np.array(
                [
                    r["accuracy"]
                    for r in rows
                    if r["condition"] == condition
                    and r["saf_rate"] == saf_rate
                ]
            ) * 100

            means.append(accs.mean())
            sds.append(accs.std(ddof=1) if len(accs) > 1 else 0.0)

        x = [s * 100 for s in SAF_LEVELS]

        ax.errorbar(
            x,
            means,
            yerr=sds,
            label=condition,
            marker="o",
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

    print(
        f"\nFigure 8 saved to "
        f"{OUT_DIR}/figure8_saf_sweep.png"
    )
    print(
        f"Done. See {OUT_DIR}/saf_sweep_summary.csv "
        f"for the Figure 8 data."
    )


if __name__ == "__main__":
    run_sweep()
