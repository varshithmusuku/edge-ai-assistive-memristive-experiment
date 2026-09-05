"""
smoke_test.py
=============
STAGE 2 of the process. Run this BEFORE experiment.py.

Does exactly ONE seed, ONE condition (Baseline), ONE SAF level (30%),
and checks each stage of the pipeline individually so that if something
breaks, you know exactly where -- instead of finding out 45 minutes into
a 5-seed x 4-condition x 100-epoch run.

Usage:
    python smoke_test.py

Every line should end in "-> OK". If ANY line errors or prints
something else, stop and fix that before running experiment.py.
"""

import sys

import numpy as np
import torch

print("=" * 60)
print("STAGE 2 SMOKE TEST")
print("=" * 60)

# ---- Check 1: MemTorch imports ----
try:
    import memtorch
    print(f"[1] MemTorch imported (version {getattr(memtorch, '__version__', 'unknown')}) -> OK")
except ImportError as e:
    print(f"[1] MemTorch import FAILED: {e}")
    print("    Fix Stage 1 first (pip install memtorch-cpu) before continuing.")
    sys.exit(1)

# ---- Check 2: VTEAM device model loads ----
try:
    from memtorch.bh.memristor import VTEAM
    print("[2] VTEAM memristor model loaded -> OK")
except ImportError as e:
    print(f"[2] VTEAM import FAILED: {e}")
    sys.exit(1)

# ---- Check 3: nonideality / SAF machinery imports ----
try:
    from memtorch.bh.nonideality import apply_nonidealities, NonIdeality
    print("[3] Nonideality / DeviceFaults machinery imported -> OK")
except ImportError as e:
    print(f"[3] Nonideality import FAILED: {e}")
    sys.exit(1)

# ---- Check 4: build a tiny model and patch it ----
from dataset import generate_dataset, CONFIG as DATA_CONFIG, CLASS_NAMES
from model import ObstacleMLP

torch.manual_seed(0)
model = ObstacleMLP()
print(f"[4] Built ObstacleMLP with architecture {[m for m in model.modules() if hasattr(m, 'out_features')]}")

try:
    import copy
    patched_model = memtorch.mn.Module.patch_model(
        copy.deepcopy(model).cpu(),
        memristor_model=VTEAM,
        memristor_model_params={},
        module_parameters_to_patch=[torch.nn.Linear],
        mapping_routine=memtorch.mn.Module.naive_map,
    )
    print("[4] Model patched into MemTorch crossbar model -> OK")
except Exception as e:
    print(f"[4] Model patching FAILED: {e}")
    sys.exit(1)

# ---- Check 5: SAF injection via apply_nonidealities ----
try:
    saf_rate = 0.30
    patched_model = apply_nonidealities(
        patched_model,
        non_idealities=[NonIdeality.DeviceFaults],
        lrs_proportion=saf_rate / 2,
        hrs_proportion=saf_rate / 2,
        electroform_proportion=0.0,
    )
    print("[5] 30% SAF injected via apply_nonidealities -> OK")
except Exception as e:
    print(f"[5] SAF injection FAILED: {e}")
    sys.exit(1)

# ---- Check 6: forward pass produces predictions ----
try:
    data = generate_dataset(master_seed=42)
    X_test, y_test = data["test"]
    X_test_t = torch.tensor(X_test[:16], dtype=torch.float32)  # just 16 samples, this is a smoke test
    patched_model.eval()
    with torch.no_grad():
        logits = patched_model(X_test_t)
        preds = logits.argmax(dim=1).numpy()
    print(f"[6] Forward pass produced predictions: {preds} -> OK")
except Exception as e:
    print(f"[6] Forward pass FAILED: {e}")
    sys.exit(1)

# ---- Check 7: confusion matrix can be built ----
try:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test[:16], preds, labels=list(range(DATA_CONFIG["num_classes"])))
    print(f"[7] Confusion matrix produced:\n{cm} -> OK")
except Exception as e:
    print(f"[7] Confusion matrix FAILED: {e}")
    sys.exit(1)

print("=" * 60)
print("ALL CHECKS PASSED. Real MemTorch/VTEAM pipeline is working.")
print("You may now proceed to Stage 3: python experiment.py")
print("=" * 60)
