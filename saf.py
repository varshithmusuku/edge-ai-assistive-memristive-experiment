"""
saf.py - corrected MemTorch/FAT/QAT implementation

Key corrections:
1. Uses the MemTorch 1.1.6 enum name: NonIdeality.
2. MemTorch calibration (tune_()) is done BEFORE DeviceFaults are injected,
   so calibration does not overwrite/compensate the faults being measured.
3. FAT/QAT training uses differentiable straight-through transformations
   instead of mutating nn.Parameter.data in-place.
4. QAT applies fake-quantization during training and the quantized weights
   are also used for QAT evaluation.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import memtorch
    from memtorch.bh.nonideality import apply_nonidealities, NonIdeality
    MEMTORCH_AVAILABLE = True
except ImportError:
    MEMTORCH_AVAILABLE = False

SAF_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
CRITICAL_SAF_LEVEL = 0.30


def make_saf_faulty_weight(weight: torch.Tensor, saf_rate: float, seed: int) -> torch.Tensor:
    """Create a faulty copy of a weight tensor without changing the Parameter."""
    if saf_rate <= 0.0:
        return weight

    rng = np.random.default_rng(seed)
    flat = weight.detach().clone().view(-1)
    numel = flat.numel()
    n_stuck = int(saf_rate * numel)

    if n_stuck <= 0:
        return weight

    idx = rng.choice(numel, size=n_stuck, replace=False)
    w_std = float(flat.std().item())
    if not np.isfinite(w_std) or w_std == 0.0:
        w_std = 1.0

    half = n_stuck // 2
    flat[idx[:half]] = -2.0 * w_std
    flat[idx[half:]] = 2.0 * w_std
    return flat.view_as(weight)


def saf_ste_weight(weight: torch.Tensor, saf_rate: float, seed: int) -> torch.Tensor:
    """
    Straight-through SAF transform:
      forward value = faulty weight
      backward gradient = gradient through the clean weight
    """
    faulty = make_saf_faulty_weight(weight, saf_rate, seed)
    return weight + (faulty - weight).detach()


def fake_quantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = 2 ** (bits - 1) - 1
    max_abs = tensor.detach().abs().max()
    if float(max_abs) == 0.0:
        return tensor
    scale = max_abs / qmax
    return torch.clamp(torch.round(tensor / scale), -qmax, qmax) * scale


def fake_quantize_ste(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Straight-through fake quantization for QAT."""
    q = fake_quantize(tensor, bits)
    return tensor + (q - tensor).detach()


def quantize_model_weights(model: nn.Module, bits: int) -> nn.Module:
    """Apply actual post-training quantization to the stored Linear weights."""
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.copy_(fake_quantize(layer.weight, bits))
    return model


def forward_condition(
    model: nn.Module,
    x: torch.Tensor,
    condition: str,
    saf_rate: float,
    quant_bits: int,
    fault_seed: int,
) -> torch.Tensor:
    """
    Forward pass used during FAT/QAT training.

    Baseline: clean floating-point weights
    FAT:       SAF-faulted weights with STE
    QAT:       SAF-faulted + fake-quantized weights with STE
    PTQ:       clean floating-point weights during training
    """
    if not hasattr(model, "net"):
        raise TypeError("Expected model with a Sequential `net` attribute.")

    out = x
    linear_id = 0

    for layer in model.net:
        if isinstance(layer, nn.Linear):
            w = layer.weight
            if condition in ("FAT", "QAT"):
                w = saf_ste_weight(
                    w,
                    saf_rate=saf_rate,
                    seed=fault_seed + 1009 * linear_id,
                )
                if condition == "QAT":
                    w = fake_quantize_ste(w, quant_bits)

            out = F.linear(out, w, layer.bias)
            linear_id += 1
        elif isinstance(layer, nn.ReLU):
            out = F.relu(out)
        else:
            out = layer(out)

    return out


def build_memtorch_eval_model(
    model: nn.Module,
    saf_rate: float,
    seed: int,
):
    """
    Real MemTorch/VTEAM evaluation model.

    Calibration is deliberately performed before DeviceFaults are applied.
    This preserves the fault effect instead of letting tuning adapt to the
    already-faulted crossbar.
    """
    if not MEMTORCH_AVAILABLE:
        raise RuntimeError(
            "MemTorch is not installed. A real MemTorch/VTEAM result cannot "
            "be generated until memtorch-cpu is available."
        )

    torch.manual_seed(seed)

    eval_model = copy.deepcopy(model).cpu()

    patched_model = memtorch.mn.Module.patch_model(
        eval_model,
        memristor_model=memtorch.bh.memristor.VTEAM,
        memristor_model_params={},
        module_parameters_to_patch=[nn.Linear],
        mapping_routine=memtorch.mn.Module.naive_map,
    )

    # Calibrate the clean crossbar representation first.
    patched_model.tune_()

    # Then inject actual MemTorch device faults.
    patched_model = apply_nonidealities(
        patched_model,
        non_idealities=[NonIdeality.DeviceFaults],
        lrs_proportion=saf_rate / 2,
        hrs_proportion=saf_rate / 2,
        electroform_proportion=0.0,
    )

    return patched_model


def build_smoketest_eval_model(model: nn.Module, saf_rate: float, seed: int):
    """Only for pipeline debugging; never use its outputs in the paper."""
    eval_model = copy.deepcopy(model)
    # Kept intentionally simple for the smoke test.
    for layer_id, layer in enumerate(eval_model.modules()):
        if isinstance(layer, nn.Linear):
            with torch.no_grad():
                faulty = make_saf_faulty_weight(
                    layer.weight, saf_rate, seed + layer_id
                )
                layer.weight.copy_(faulty)
    return eval_model
