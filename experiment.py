"""
experiment.py - corrected reproducible experiment runner.

Conditions:
  Baseline = clean floating-point training
  PTQ      = clean training, then post-training weight quantization
  FAT      = SAF-aware training using a differentiable straight-through fault transform
  QAT      = SAF-aware + quantization-aware training using straight-through transforms

The dataset/split is fixed across all conditions and seeds.
"""

import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from dataset import generate_dataset, CONFIG as DATA_CONFIG, CLASS_NAMES
from model import ObstacleMLP
from saf import (
    forward_condition,
    quantize_model_weights,
    build_memtorch_eval_model,
    CRITICAL_SAF_LEVEL,
    MEMTORCH_AVAILABLE,
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
OUT_DIR = "results"


def train_model(model, X_train, y_train, condition: str, seed: int, cfg=TRAIN_CONFIG):
    """Train one model for one condition and one seed."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    n = len(y_train_t)

    model.train()

    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(n)

        for batch_id, i in enumerate(range(0, n, cfg["batch_size"])):
            idx = perm[i:i + cfg["batch_size"]]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad(set_to_none=True)

            fault_seed = seed * 1_000_000 + epoch * 10_000 + batch_id

            out = forward_condition(
                model,
                xb,
                condition=condition,
                saf_rate=cfg["train_saf_rate"],
                quant_bits=cfg["quant_bits"],
                fault_seed=fault_seed,
            )

            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

    return model


def run_condition_eval(trained_model, X_test, y_test, condition: str, seed: int):
    """Evaluate the final model using the real MemTorch/VTEAM path."""
    if condition in ("PTQ", "QAT"):
        quantize_model_weights(trained_model, TRAIN_CONFIG["quant_bits"])

    if not MEMTORCH_AVAILABLE:
        raise RuntimeError(
            "MEMTORCH_AVAILABLE=False. Refusing to generate paper results."
        )

    eval_model = build_memtorch_eval_model(
        trained_model,
        CRITICAL_SAF_LEVEL,
        seed + 100_000,
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


def log_environment(out_dir):
    lines = [
        f"python: {sys.version}",
        f"torch: {torch.__version__}",
    ]

    try:
        import memtorch
        lines.append(f"memtorch: {getattr(memtorch, '__version__', 'unknown')}")
    except ImportError:
        lines.append("memtorch: NOT INSTALLED")

    lines.extend(
        [
            f"numpy: {np.__version__}",
            f"dataset_master_seed: {DATASET_MASTER_SEED}",
            f"seeds: {SEEDS}",
            f"classes: {CLASS_NAMES}",
            f"train_config: {TRAIN_CONFIG}",
            f"memtorch_available: {MEMTORCH_AVAILABLE}",
        ]
    )

    os.makedirs(out_dir, exist_ok=True)

    with open(
        os.path.join(out_dir, "environment.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


def run_all():
    if not MEMTORCH_AVAILABLE:
        raise RuntimeError(
            "MemTorch is unavailable. Fix the environment before running "
            "the experiment."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    log_environment(OUT_DIR)

    fixed_data = generate_dataset(master_seed=DATASET_MASTER_SEED)

    print(
        f"Dataset: {fixed_data['total_samples']} total samples "
        f"(classes: {CLASS_NAMES})"
    )

    rows = []
    cms = {
        c: np.zeros(
            (DATA_CONFIG["num_classes"], DATA_CONFIG["num_classes"]),
            dtype=int,
        )
        for c in CONDITIONS
    }

    for condition in CONDITIONS:
        for seed in SEEDS:
            data = (
                generate_dataset(master_seed=DATASET_MASTER_SEED + seed)
                if REGENERATE_DATA_PER_SEED
                else fixed_data
            )

            X_train, y_train = data["train"]
            X_test, y_test = data["test"]

            # IMPORTANT: seed BEFORE constructing the model.
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = ObstacleMLP()
            model = train_model(
                model,
                X_train,
                y_train,
                condition,
                seed,
            )

            acc, cm = run_condition_eval(
                model,
                X_test,
                y_test,
                condition,
                seed,
            )

            cms[condition] += cm
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "accuracy": acc,
                }
            )

            print(
                f"[{condition}] seed={seed} "
                f"accuracy={acc * 100:.1f}%"
            )

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
        writer.writerows(rows)

    with open(
        os.path.join(OUT_DIR, "summary.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                "n_runs",
                "mean_accuracy_pct",
                "sample_sd_accuracy_pct",
            ]
        )

        for condition in CONDITIONS:
            accs = np.array(
                [
                    r["accuracy"]
                    for r in rows
                    if r["condition"] == condition
                ]
            ) * 100

            sd = accs.std(ddof=1) if len(accs) > 1 else 0.0

            writer.writerow(
                [
                    condition,
                    len(accs),
                    f"{accs.mean():.1f}",
                    f"{sd:.1f}",
                ]
            )

            print(
                f"TABLE X ROW -> {condition}: "
                f"{accs.mean():.1f}% +/- {sd:.1f}% "
                f"(N={len(accs)} runs, seeds={SEEDS}, sample SD)"
            )

    try:
        import matplotlib.pyplot as plt

        for condition, cm in cms.items():
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
                f"(summed over {len(SEEDS)} runs)"
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

    except ImportError:
        print("matplotlib not installed; .npy confusion matrices were still saved.")

    print(f"\nDone. Results in ./{OUT_DIR}/")


if __name__ == "__main__":
    run_all()
