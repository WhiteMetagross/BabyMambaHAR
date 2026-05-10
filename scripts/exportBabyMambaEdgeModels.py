from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
MODEL_ROOT = REPO_ROOT / "models"
DEFAULT_EXPORT_ROOT = REPO_ROOT / "Pico2Models"

sys.path.insert(0, str(REPO_ROOT))
# Crossover package is imported from the repository root.

from ciBabyMambaHar.models import CiBabyMambaHar
from ciBabyMambaHar.models.ciBabyMamba import CI_BABYMAMBA_HAR_CONFIG
from ciBabyMambaHar.models.ciBabyMambaBlock import PureSelectiveScan
from crossoverBiDirBabyMambaHar.models import (
    CROSSOVER_BIDIR_BABYMAMBA_CONFIG,
    CrossoverBiDirBabyMambaHar,
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    folder: str
    display_name: str
    in_channels: int
    seq_len: int
    num_classes: int
    class_names: Sequence[str]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "ucihar": DatasetSpec(
        key="ucihar",
        folder="uciHar",
        display_name="UCI-HAR",
        in_channels=9,
        seq_len=128,
        num_classes=6,
        class_names=(
            "Walking",
            "Walking Upstairs",
            "Walking Downstairs",
            "Sitting",
            "Standing",
            "Laying",
        ),
    ),
    "motionsense": DatasetSpec(
        key="motionsense",
        folder="motionSense",
        display_name="MotionSense",
        in_channels=6,
        seq_len=128,
        num_classes=6,
        class_names=(
            "Downstairs",
            "Upstairs",
            "Walking",
            "Jogging",
            "Sitting",
            "Standing",
        ),
    ),
    "wisdm": DatasetSpec(
        key="wisdm",
        folder="wisdm",
        display_name="WISDM",
        in_channels=3,
        seq_len=128,
        num_classes=6,
        class_names=(
            "Walking",
            "Jogging",
            "Upstairs",
            "Downstairs",
            "Sitting",
            "Standing",
        ),
    ),
    "pamap2": DatasetSpec(
        key="pamap2",
        folder="pamap2",
        display_name="PAMAP2",
        in_channels=19,
        seq_len=128,
        num_classes=12,
        class_names=(
            "Lying",
            "Sitting",
            "Standing",
            "Walking",
            "Running",
            "Cycling",
            "Nordic Walking",
            "Ascending Stairs",
            "Descending Stairs",
            "Vacuum Cleaning",
            "Ironing",
            "Rope Jumping",
        ),
    ),
    "opportunity": DatasetSpec(
        key="opportunity",
        folder="opportunity",
        display_name="Opportunity",
        in_channels=79,
        seq_len=128,
        num_classes=5,
        class_names=("Null", "Stand", "Walk", "Sit", "Lie"),
    ),
    "unimib": DatasetSpec(
        key="unimib",
        folder="unimib",
        display_name="UniMiB-SHAR",
        in_channels=3,
        seq_len=128,
        num_classes=9,
        class_names=tuple(f"Activity_{idx}" for idx in range(9)),
    ),
    "skoda": DatasetSpec(
        key="skoda",
        folder="skoda",
        display_name="Skoda",
        in_channels=30,
        seq_len=98,
        num_classes=11,
        class_names=("Null",) + tuple(f"Gesture_{idx}" for idx in range(10)),
    ),
    "daphnet": DatasetSpec(
        key="daphnet",
        folder="daphnet",
        display_name="Daphnet",
        in_channels=9,
        seq_len=64,
        num_classes=2,
        class_names=("No Freeze", "Freeze"),
    ),
}


VARIANT_CHOICES = ("crossoverBiDirBabyMambaHar", "ciBabyMambaHar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BabyMamba checkpoints into Pico 2 friendly C headers."
    )
    parser.add_argument(
        "--variant",
        choices=VARIANT_CHOICES,
        default="crossoverBiDirBabyMambaHar",
        help="BabyMamba family to export.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset keys or 'all'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed to export. If omitted, the best F1 seed from summary.json is used.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Test sample index used for the exported parity fixture.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="Root directory for exported BabyMamba bundles.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if a requested checkpoint is missing.",
    )
    return parser.parse_args()


def parse_csv_or_all(raw: str, choices: Sequence[str]) -> List[str]:
    if raw.strip().lower() == "all":
        return list(choices)
    items = [part.strip().lower() for part in raw.split(",") if part.strip()]
    invalid = [item for item in items if item not in choices]
    if invalid:
        raise ValueError(f"Invalid datasets {invalid}. Choices: {choices}")
    return items


def conv1d_output_len(length: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return ((length + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1


def find_best_seed(summary_path: Path) -> int:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No results found in {summary_path}")
    best = max(results, key=lambda item: float(item.get("testF1", item.get("bestF1", -1.0))))
    return int(best["seed"])


def find_checkpoint_for_crossover(dataset_key: str, preferred_seed: Optional[int]) -> Tuple[Path, int, Path]:
    zoo_dir = MODEL_ROOT / "crossoverBiDirBabyMambaHar" / dataset_key
    if zoo_dir.exists():
        checkpoint_candidates = sorted(zoo_dir.glob("best*_seed*.pt"))
        if checkpoint_candidates:
            checkpoint_path = checkpoint_candidates[-1]
            seed_match = checkpoint_path.stem.split("seed")[-1]
            seed = preferred_seed if preferred_seed is not None else int(seed_match)
            return checkpoint_path, seed, zoo_dir

    spec = DATASET_SPECS[dataset_key]
    dataset_root = RESULTS_ROOT / "training" / spec.folder
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing training folder for {dataset_key}: {dataset_root}")

    candidate_dirs = []
    for child in dataset_root.iterdir():
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        if summary_path.exists() and any(child.glob("best_model_seed*.pt")):
            candidate_dirs.append(child)
    if not candidate_dirs:
        raise FileNotFoundError(f"No deployable Crossover-BiDir checkpoints found for {dataset_key}")

    candidate_dirs.sort(key=lambda path: path.name)
    run_dir = candidate_dirs[-1]
    seed = preferred_seed if preferred_seed is not None else find_best_seed(run_dir / "summary.json")
    checkpoint_path = run_dir / f"best_model_seed{seed}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint_path}")
    return checkpoint_path, seed, run_dir


def find_checkpoint_for_ci(dataset_key: str, preferred_seed: Optional[int]) -> Tuple[Path, int, Path]:
    zoo_dir = MODEL_ROOT / "ciBabyMambaHar" / dataset_key
    if zoo_dir.exists():
        checkpoint_candidates = sorted(zoo_dir.glob("best*_seed*.pt"))
        if checkpoint_candidates:
            checkpoint_path = checkpoint_candidates[-1]
            seed_match = checkpoint_path.stem.split("seed")[-1]
            seed = preferred_seed if preferred_seed is not None else int(seed_match)
            return checkpoint_path, seed, zoo_dir

    dataset_root = RESULTS_ROOT / "training" / "ciBabyMambaHar"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing CiBabyMambaHar training root: {dataset_root}")

    candidate_runs: List[Tuple[Path, Path]] = []
    for checkpoint in dataset_root.rglob("best_model_seed*.pt"):
        parent = checkpoint.parent
        if parent.name.startswith("202") and parent.parent.name.lower() == dataset_key:
            candidate_runs.append((parent, checkpoint))

    if not candidate_runs:
        raise FileNotFoundError(
            f"No saved CI-BabyMamba checkpoint files were found for {dataset_key} under {dataset_root}"
        )

    def score(item: Tuple[Path, Path]) -> Tuple[int, float, str]:
        run_dir, checkpoint_path = item
        run_group = run_dir.parent.parent.name.lower()
        is_smoke = 1 if "smoke" in run_group else 0
        return (-is_smoke, checkpoint_path.stat().st_mtime, str(run_dir))

    run_dir, _ = max(candidate_runs, key=score)
    seed = preferred_seed if preferred_seed is not None else find_best_seed(run_dir / "summary.json")
    checkpoint_path = run_dir / f"best_model_seed{seed}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint_path}")
    return checkpoint_path, seed, run_dir


def extract_state_dict(payload: Dict[str, object]) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "modelStateDict", "state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload  # type: ignore[return-value]


def state_dict_uses_fallback_style(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(".ssm.ALog" in key or ".ssm.inProj.weight" in key for key in state_dict.keys())


def get_first_tensor(
    state_dict: Dict[str, torch.Tensor],
    candidates: Sequence[str],
) -> torch.Tensor:
    for key in candidates:
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise KeyError(f"None of the state_dict keys were found: {candidates}")


def build_model(variant: str, dataset_key: str, dropout: float = 0.0, force_fallback: bool = False):
    spec = DATASET_SPECS[dataset_key]
    if variant == "crossoverBiDirBabyMambaHar":
        return CrossoverBiDirBabyMambaHar(
            numClasses=spec.num_classes,
            inChannels=spec.in_channels,
            seqLen=spec.seq_len,
            dropout=dropout,
        )
    if not force_fallback:
        return CiBabyMambaHar(
            numClasses=spec.num_classes,
            inChannels=spec.in_channels,
            seqLen=spec.seq_len,
            dropout=dropout,
            channelIndependent=True,
            bidirectional=True,
            useGatedAttention=True,
        )

    import ciBabyMambaHar.models.ciBabyMambaBlock as ci_baby_mamba_block

    original = ci_baby_mamba_block.MAMBA_AVAILABLE
    ci_baby_mamba_block.MAMBA_AVAILABLE = False
    try:
        return CiBabyMambaHar(
            numClasses=spec.num_classes,
            inChannels=spec.in_channels,
            seqLen=spec.seq_len,
            dropout=dropout,
            channelIndependent=True,
            bidirectional=True,
            useGatedAttention=True,
        )
    finally:
        ci_baby_mamba_block.MAMBA_AVAILABLE = original


def load_dataset_test_samples(dataset_key: str) -> List[Tuple[np.ndarray, int]]:
    spec = DATASET_SPECS[dataset_key]
    root = REPO_ROOT / "datasets"
    if dataset_key == "ucihar":
        from ciBabyMambaHar.data.uciHar import UciHarDataset

        ds = UciHarDataset(str(root / "UCI HAR Dataset"), split="test")
    elif dataset_key == "motionsense":
        from ciBabyMambaHar.data.motionSense import MotionSenseDataset

        ds = MotionSenseDataset(str(root / "motion-sense-master"), split="test", windowSize=spec.seq_len)
    elif dataset_key == "wisdm":
        from ciBabyMambaHar.data.wisdm import WisdmDataset

        ds = WisdmDataset(str(root / "WISDM_ar_v1.1"), split="test", windowSize=spec.seq_len)
    elif dataset_key == "pamap2":
        from ciBabyMambaHar.data.pamap2 import Pamap2Dataset

        ds = Pamap2Dataset(
            str(root / "PAMAP2_Dataset"),
            split="test",
            windowSize=spec.seq_len,
            applyFilter=True,
            filterCutoff=10.0,
            useRobustScaling=True,
        )
    elif dataset_key == "opportunity":
        from ciBabyMambaHar.data.opportunity import OpportunityDataset

        ds = OpportunityDataset(str(root / "Opportunity"), split="test", task="locomotion")
    elif dataset_key == "unimib":
        from ciBabyMambaHar.data.unimib import UniMiBSHARDataset

        ds = UniMiBSHARDataset(str(root / "UniMiB-SHAR"), split="test", task="adl")
    elif dataset_key == "skoda":
        from ciBabyMambaHar.data.skoda import SkodaDataset

        ds = SkodaDataset(str(root / "Skoda"), split="test", windowSize=spec.seq_len)
    elif dataset_key == "daphnet":
        from ciBabyMambaHar.data.daphnet import DaphnetDataset

        ds = DaphnetDataset(str(root / "Daphnet"), split="test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")

    samples: List[Tuple[np.ndarray, int]] = []
    for idx in range(len(ds)):
        item = ds[idx]
        if isinstance(item, tuple) and len(item) == 2:
            sample, label = item
        else:
            raise ValueError(f"Unexpected dataset item for {dataset_key}: {type(item)}")
        sample_np = np.asarray(sample, dtype=np.float32)
        if sample_np.ndim == 2:
            # Dataset loaders usually return [T, C].
            pass
        elif sample_np.ndim == 3 and sample_np.shape[0] == 1:
            sample_np = sample_np[0]
        else:
            raise ValueError(f"Unexpected sample shape for {dataset_key}: {sample_np.shape}")
        samples.append((sample_np, int(label)))
    return samples


def fold_bn_into_conv1d(
    conv_weight: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    scale = bn_weight / torch.sqrt(running_var + eps)
    folded_weight = conv_weight * scale.reshape(-1, 1, 1)
    folded_bias = bn_bias - scale * running_mean
    return folded_weight.detach().cpu().numpy().astype(np.float32), folded_bias.detach().cpu().numpy().astype(np.float32)


def fold_bn_into_linear_out(
    linear_weight: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    scale = bn_weight / torch.sqrt(running_var + eps)
    folded_weight = linear_weight * scale.reshape(-1, 1)
    folded_bias = bn_bias - scale * running_mean
    return folded_weight.detach().cpu().numpy().astype(np.float32), folded_bias.detach().cpu().numpy().astype(np.float32)


def interpolate_pos_embed(pos_embed: torch.Tensor, target_len: int) -> np.ndarray:
    if pos_embed.shape[1] == target_len:
        return pos_embed.detach().cpu().numpy().astype(np.float32)
    interpolated = F.interpolate(
        pos_embed.transpose(1, 2),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    return interpolated.detach().cpu().numpy().astype(np.float32)


def softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def layernorm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    return normalized * weight + bias


def conv1d_same_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, padding: int) -> np.ndarray:
    out_channels, in_channels, kernel = weight.shape
    seq_len = x.shape[1]
    padded = np.pad(x, ((0, 0), (padding, padding)), mode="constant")
    out = np.zeros((out_channels, seq_len), dtype=np.float32)
    for oc in range(out_channels):
        for t in range(seq_len):
            acc = float(bias[oc])
            window = padded[:, t : t + kernel]
            acc += float(np.sum(window * weight[oc]))
            out[oc, t] = acc
    return out


def depthwise_conv_stride_np(x: np.ndarray, weight: np.ndarray, stride: int, padding: int) -> np.ndarray:
    channels, kernel = weight.shape
    seq_len = x.shape[1]
    padded = np.pad(x, ((0, 0), (padding, padding)), mode="constant")
    out_len = conv1d_output_len(seq_len, kernel, stride, padding)
    out = np.zeros((channels, out_len), dtype=np.float32)
    for c in range(channels):
        for out_idx in range(out_len):
            start = out_idx * stride
            out[c, out_idx] = float(np.dot(padded[c, start : start + kernel], weight[c]))
    return out


def pointwise_conv_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    out_channels, in_channels = weight.shape
    seq_len = x.shape[1]
    out = np.zeros((out_channels, seq_len), dtype=np.float32)
    for t in range(seq_len):
        out[:, t] = weight @ x[:, t] + bias
    return out


def gated_attention_np(
    seq_in: np.ndarray,
    projection_weight: np.ndarray,
    projection_bias: np.ndarray,
    context: np.ndarray,
) -> np.ndarray:
    gated = np.tanh(seq_in @ projection_weight.T + projection_bias)
    scores = gated @ context
    scores = scores - np.max(scores)
    alpha = np.exp(scores)
    alpha_sum = np.sum(alpha) + 1e-6
    alpha = alpha / alpha_sum
    return alpha @ seq_in


def selective_scan_forward_np(
    seq_in: np.ndarray,
    layer: Dict[str, np.ndarray],
) -> np.ndarray:
    seq_len, d_model = seq_in.shape
    d_inner = int(layer["d_inner"])
    d_state = int(layer["d_state"])
    dt_rank = int(layer["dt_rank"])
    chunk_size = 32

    xz = seq_in @ layer["inProj"].T
    x_part = xz[:, :d_inner]
    z_part = xz[:, d_inner:]

    history = np.zeros((d_inner, int(layer["d_conv"])), dtype=np.float32)
    x_act = np.zeros((seq_len, d_inner), dtype=np.float32)
    for idx in range(seq_len):
        history[:, :-1] = history[:, 1:]
        history[:, -1] = x_part[idx]
        conv = np.sum(history * layer["conv1dWeight"], axis=1) + layer["conv1dBias"]
        x_act[idx] = silu_np(conv)

    proj = x_act @ layer["xProj"].T
    dt_raw = proj[:, :dt_rank]
    B = proj[:, dt_rank : dt_rank + d_state]
    C = proj[:, dt_rank + d_state :]

    dt = dt_raw @ layer["dtProjWeight"].T + layer["dtProjBias"]
    dt = softplus_np(dt)

    A = -np.exp(layer["ALog"]).astype(np.float32)
    D = layer["D"].astype(np.float32)

    delta_a = np.exp(dt[:, :, np.newaxis] * A[np.newaxis, :, :]).astype(np.float32)
    delta_bx = (
        dt[:, :, np.newaxis]
        * B[:, np.newaxis, :]
        * x_act[:, :, np.newaxis]
    ).astype(np.float32)

    h = np.zeros((d_inner, d_state), dtype=np.float32)
    outputs: List[np.ndarray] = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_a = delta_a[start:end]
        chunk_bx = delta_bx[start:end]
        chunk_c = C[start:end]
        chunk_x = x_act[start:end]

        log_a = np.log(np.clip(chunk_a, 1e-6, None)).astype(np.float32)
        cum_log_a = np.cumsum(log_a, axis=0, dtype=np.float32)
        cum_a = np.exp(np.clip(cum_log_a, None, 20.0)).astype(np.float32)

        h_init_contrib = h[np.newaxis, :, :] * cum_a
        inv_cum_a = np.exp(-np.clip(cum_log_a, -20.0, 20.0)).astype(np.float32)
        scaled_bx = chunk_bx * inv_cum_a
        cum_scaled_bx = np.cumsum(scaled_bx, axis=0, dtype=np.float32)
        h_chunk = h_init_contrib + (cum_a * cum_scaled_bx)
        h = h_chunk[-1]

        y_chunk = np.einsum("ldi,li->ld", h_chunk, chunk_c, optimize=True).astype(np.float32)
        y_chunk = y_chunk + (chunk_x * D)
        outputs.append(y_chunk)

    y = np.concatenate(outputs, axis=0)
    y = y * silu_np(z_part)
    return (y @ layer["outProj"].T).astype(np.float32)


def selective_scan_sequential_np(
    seq_in: np.ndarray,
    layer: Dict[str, np.ndarray],
    reverse: bool,
) -> np.ndarray:
    seq_len, d_model = seq_in.shape
    d_inner = int(layer["d_inner"])
    d_state = int(layer["d_state"])
    d_conv = int(layer["d_conv"])

    order = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    outputs = np.zeros((seq_len, d_model), dtype=np.float32)
    state = np.zeros((d_inner, d_state), dtype=np.float32)
    history = np.zeros((d_inner, d_conv), dtype=np.float32)
    A = -np.exp(layer["ALog"]).astype(np.float32)
    D = layer["D"]

    for idx in order:
        token = seq_in[idx]
        xz = layer["inProj"] @ token
        x_part = xz[:d_inner]
        z_part = xz[d_inner:]

        history[:, :-1] = history[:, 1:]
        history[:, -1] = x_part

        conv = np.sum(history * layer["conv1dWeight"], axis=1) + layer["conv1dBias"]
        x_act = silu_np(conv)

        proj = layer["xProj"] @ x_act
        dt_raw = proj[: int(layer["dt_rank"])]
        B = proj[int(layer["dt_rank"]) : int(layer["dt_rank"]) + d_state]
        C = proj[int(layer["dt_rank"]) + d_state :]

        dt = softplus_np(layer["dtProjWeight"] @ dt_raw + layer["dtProjBias"])
        y_inner = np.zeros((d_inner,), dtype=np.float32)
        gated = silu_np(z_part)

        for di in range(d_inner):
            running = D[di] * x_act[di]
            for n in range(d_state):
                decay = math.exp(float(dt[di] * A[di, n]))
                state[di, n] = decay * state[di, n] + float(dt[di] * B[n] * x_act[di])
                running += state[di, n] * C[n]
            y_inner[di] = running * gated[di]

        outputs[idx] = layer["outProj"] @ y_inner

    return outputs


def selective_scan_np(
    seq_in: np.ndarray,
    layer: Dict[str, np.ndarray],
    reverse: bool,
) -> np.ndarray:
    scan_impl = layer.get("scanImpl", "sequential_recurrence")
    if scan_impl == "fallback_chunked":
        if reverse:
            flipped = np.flip(seq_in, axis=0).copy()
            return np.flip(selective_scan_forward_np(flipped, layer), axis=0).copy()
        return selective_scan_forward_np(seq_in, layer)
    return selective_scan_sequential_np(seq_in, layer, reverse)


def run_crossover_reference(
    arrays: Dict[str, object],
    sample: np.ndarray,
) -> np.ndarray:
    config = arrays["config"]
    assert isinstance(config, dict)
    stem_weight = arrays["stemWeight"]
    stem_bias = arrays["stemBias"]
    patch_depthwise = arrays["patchDepthwise"]
    patch_pointwise = arrays["patchPointwise"]
    patch_bias = arrays["patchBias"]
    pos_embed = arrays["posEmbed"]
    layers = arrays["layers"]
    head_norm_weight = arrays["headNormWeight"]
    head_norm_bias = arrays["headNormBias"]
    head_linear = arrays["headLinearWeight"]
    head_linear_bias = arrays["headLinearBias"]

    x = sample.T.astype(np.float32)
    x = conv1d_same_np(x, stem_weight, stem_bias, padding=config["stem_padding"])
    x = silu_np(x)

    x = depthwise_conv_stride_np(
        x,
        patch_depthwise,
        stride=config["patch_stride"],
        padding=config["patch_padding"],
    )
    x = pointwise_conv_np(x, patch_pointwise, patch_bias)
    x = silu_np(x)
    x = x.T
    x = x + pos_embed[0]

    for layer in layers:
        pre = layernorm_np(x, layer["preNormWeight"], layer["preNormBias"])
        fwd = selective_scan_np(pre, layer, reverse=False)
        bwd = selective_scan_np(pre, layer, reverse=True)
        x = layernorm_np(x + fwd + bwd, layer["postNormWeight"], layer["postNormBias"])

    x = x.mean(axis=0)
    x = layernorm_np(x[np.newaxis, :], head_norm_weight, head_norm_bias)[0]
    logits = head_linear @ x + head_linear_bias
    return logits.astype(np.float32)


def run_ci_reference(
    arrays: Dict[str, object],
    sample: np.ndarray,
) -> np.ndarray:
    config = arrays["config"]
    assert isinstance(config, dict)
    stem_weight = np.asarray(arrays["stemWeight"], dtype=np.float32)
    stem_bias = np.asarray(arrays["stemBias"], dtype=np.float32)
    patch_depthwise = np.asarray(arrays["patchDepthwise"], dtype=np.float32)
    patch_pointwise = np.asarray(arrays["patchPointwise"], dtype=np.float32)
    patch_bias = np.asarray(arrays["patchBias"], dtype=np.float32)
    pos_embed = np.asarray(arrays["posEmbed"], dtype=np.float32)
    layers = arrays["layers"]
    assert isinstance(layers, list)
    head_norm_weight = np.asarray(arrays["headNormWeight"], dtype=np.float32)
    head_norm_bias = np.asarray(arrays["headNormBias"], dtype=np.float32)
    head_linear = np.asarray(arrays["headLinearWeight"], dtype=np.float32)
    head_linear_bias = np.asarray(arrays["headLinearBias"], dtype=np.float32)
    attention_weight = np.asarray(arrays["gatedProjectionWeight"], dtype=np.float32)
    attention_bias = np.asarray(arrays["gatedProjectionBias"], dtype=np.float32)
    attention_context = np.asarray(arrays["gatedContext"], dtype=np.float32)

    channel_outputs: List[np.ndarray] = []
    for channel_idx in range(sample.shape[1]):
        x = sample[:, channel_idx].astype(np.float32)[np.newaxis, :]
        x = conv1d_same_np(x, stem_weight, stem_bias, padding=config["stem_padding"])
        x = silu_np(x)

        x = depthwise_conv_stride_np(
            x,
            patch_depthwise,
            stride=config["patch_stride"],
            padding=config["patch_padding"],
        )
        x = pointwise_conv_np(x, patch_pointwise, patch_bias)
        x = silu_np(x)
        x = x.T
        x = x + pos_embed[0]

        for layer in layers:
            assert isinstance(layer, dict)
            pre = layernorm_np(x, layer["preNormWeight"], layer["preNormBias"])
            fwd = selective_scan_np(pre, layer, reverse=False)
            bwd = selective_scan_np(pre, layer, reverse=True)
            x = layernorm_np(x + fwd + bwd, layer["postNormWeight"], layer["postNormBias"])

        channel_outputs.append(gated_attention_np(x, attention_weight, attention_bias, attention_context))

    fused = np.mean(np.stack(channel_outputs, axis=0), axis=0)
    fused = layernorm_np(fused[np.newaxis, :], head_norm_weight, head_norm_bias)[0]
    logits = head_linear @ fused + head_linear_bias
    return logits.astype(np.float32)


def numpy_arrays_for_crossover(model: CrossoverBiDirBabyMambaHar) -> Dict[str, object]:
    state_dict = model.state_dict()
    scan_impl = "sequential_recurrence"
    stem_weight, stem_bias = fold_bn_into_conv1d(
        state_dict["stem.0.weight"],
        state_dict["stem.1.weight"],
        state_dict["stem.1.bias"],
        state_dict["stem.1.running_mean"],
        state_dict["stem.1.running_var"],
    )
    patch_pointwise, patch_bias = fold_bn_into_linear_out(
        state_dict["patchPointwise.weight"].squeeze(-1),
        state_dict["patchNorm.running_mean"],
        state_dict["patchNorm.running_var"],
        state_dict["patchNorm.weight"],
        state_dict["patchNorm.bias"],
    )
    patch_out_len = conv1d_output_len(
        model.seqLen,
        model.patchKernel,
        model.patchStride,
        model.patchKernel // 4,
    )
    pos_embed = interpolate_pos_embed(state_dict["posEmbed"], patch_out_len)

    layers: List[Dict[str, np.ndarray]] = []
    for layer_idx in range(model.nLayers):
        prefix = f"mambaLayers.{layer_idx}"
        layer_arrays: Dict[str, np.ndarray] = {
            "preNormWeight": state_dict[f"{prefix}.preNorm.weight"].detach().cpu().numpy().astype(np.float32),
            "preNormBias": state_dict[f"{prefix}.preNorm.bias"].detach().cpu().numpy().astype(np.float32),
            "postNormWeight": state_dict[f"{prefix}.postNorm.weight"].detach().cpu().numpy().astype(np.float32),
            "postNormBias": state_dict[f"{prefix}.postNorm.bias"].detach().cpu().numpy().astype(np.float32),
            "ALog": get_first_tensor(state_dict, [f"{prefix}.ssm.ALog", f"{prefix}.ssm.A_log"]).detach().cpu().numpy().astype(np.float32),
            "D": get_first_tensor(state_dict, [f"{prefix}.ssm.D"]).detach().cpu().numpy().astype(np.float32),
            "inProj": get_first_tensor(state_dict, [f"{prefix}.ssm.inProj.weight", f"{prefix}.ssm.in_proj.weight"]).detach().cpu().numpy().astype(np.float32),
            "conv1dWeight": get_first_tensor(state_dict, [f"{prefix}.ssm.conv1d.weight"]).squeeze(1).detach().cpu().numpy().astype(np.float32),
            "conv1dBias": get_first_tensor(state_dict, [f"{prefix}.ssm.conv1d.bias"]).detach().cpu().numpy().astype(np.float32),
            "xProj": get_first_tensor(state_dict, [f"{prefix}.ssm.xProj.weight", f"{prefix}.ssm.x_proj.weight"]).detach().cpu().numpy().astype(np.float32),
            "dtProjWeight": get_first_tensor(state_dict, [f"{prefix}.ssm.dtProj.weight", f"{prefix}.ssm.dt_proj.weight"]).detach().cpu().numpy().astype(np.float32),
            "dtProjBias": get_first_tensor(state_dict, [f"{prefix}.ssm.dtProj.bias", f"{prefix}.ssm.dt_proj.bias"]).detach().cpu().numpy().astype(np.float32),
            "outProj": get_first_tensor(state_dict, [f"{prefix}.ssm.outProj.weight", f"{prefix}.ssm.out_proj.weight"]).detach().cpu().numpy().astype(np.float32),
        }
        layer_arrays["d_inner"] = np.array(model.dModel * model.expand, dtype=np.int32)
        layer_arrays["d_state"] = np.array(model.dState, dtype=np.int32)
        layer_arrays["d_conv"] = np.array(model.dConv, dtype=np.int32)
        layer_arrays["dt_rank"] = np.array(model.dtRank, dtype=np.int32)
        layer_arrays["scanImpl"] = scan_impl
        layers.append(layer_arrays)

    return {
        "config": {
            "variant": "crossoverBiDirBabyMambaHar",
            "scan_impl": scan_impl,
            "seq_len": model.seqLen,
            "in_channels": model.inChannels,
            "num_classes": model.numClasses,
            "d_model": model.dModel,
            "d_state": model.dState,
            "expand": model.expand,
            "d_inner": model.dModel * model.expand,
            "dt_rank": model.dtRank,
            "d_conv": model.dConv,
            "n_layers": model.nLayers,
            "stem_kernel": model.stemKernel,
            "stem_padding": model.stemKernel // 2,
            "patch_kernel": model.patchKernel,
            "patch_stride": model.patchStride,
            "patch_padding": model.patchKernel // 4,
            "patch_out_len": patch_out_len,
        },
        "stemWeight": stem_weight,
        "stemBias": stem_bias,
        "patchDepthwise": state_dict["patchDepthwise.weight"].squeeze(1).detach().cpu().numpy().astype(np.float32),
        "patchPointwise": patch_pointwise,
        "patchBias": patch_bias,
        "posEmbed": pos_embed,
        "layers": layers,
        "headNormWeight": state_dict["headNorm.weight"].detach().cpu().numpy().astype(np.float32),
        "headNormBias": state_dict["headNorm.bias"].detach().cpu().numpy().astype(np.float32),
        "headLinearWeight": state_dict["headLinear.weight"].detach().cpu().numpy().astype(np.float32),
        "headLinearBias": state_dict["headLinear.bias"].detach().cpu().numpy().astype(np.float32),
    }


def numpy_arrays_for_ci(model: CiBabyMambaHar) -> Dict[str, object]:
    state_dict = model.state_dict()
    scan_impl = "fallback_chunked" if isinstance(model.mambaLayers[0].ssm, PureSelectiveScan) else "sequential_recurrence"
    stem_weight, stem_bias = fold_bn_into_conv1d(
        state_dict["ciStem.conv.weight"],
        state_dict["ciStem.norm.weight"],
        state_dict["ciStem.norm.bias"],
        state_dict["ciStem.norm.running_mean"],
        state_dict["ciStem.norm.running_var"],
    )
    patch_pointwise, patch_bias = fold_bn_into_linear_out(
        state_dict["patchPointwise.weight"].squeeze(-1),
        state_dict["patchNorm.running_mean"],
        state_dict["patchNorm.running_var"],
        state_dict["patchNorm.weight"],
        state_dict["patchNorm.bias"],
    )
    patch_out_len = conv1d_output_len(
        model.seqLen,
        model.patchKernel,
        model.patchStride,
        model.patchKernel // 4,
    )
    pos_embed = interpolate_pos_embed(state_dict["posEmbed"], patch_out_len)

    layers: List[Dict[str, np.ndarray]] = []
    for layer_idx in range(model.nLayers):
        prefix = f"mambaLayers.{layer_idx}"
        layer_arrays: Dict[str, np.ndarray] = {
            "preNormWeight": state_dict[f"{prefix}.preNorm.weight"].detach().cpu().numpy().astype(np.float32),
            "preNormBias": state_dict[f"{prefix}.preNorm.bias"].detach().cpu().numpy().astype(np.float32),
            "postNormWeight": state_dict[f"{prefix}.postNorm.weight"].detach().cpu().numpy().astype(np.float32),
            "postNormBias": state_dict[f"{prefix}.postNorm.bias"].detach().cpu().numpy().astype(np.float32),
            "ALog": get_first_tensor(state_dict, [f"{prefix}.ssm.ALog", f"{prefix}.ssm.A_log"]).detach().cpu().numpy().astype(np.float32),
            "D": get_first_tensor(state_dict, [f"{prefix}.ssm.D"]).detach().cpu().numpy().astype(np.float32),
            "inProj": get_first_tensor(state_dict, [f"{prefix}.ssm.inProj.weight", f"{prefix}.ssm.in_proj.weight"]).detach().cpu().numpy().astype(np.float32),
            "conv1dWeight": get_first_tensor(state_dict, [f"{prefix}.ssm.conv1d.weight"]).squeeze(1).detach().cpu().numpy().astype(np.float32),
            "conv1dBias": get_first_tensor(state_dict, [f"{prefix}.ssm.conv1d.bias"]).detach().cpu().numpy().astype(np.float32),
            "xProj": get_first_tensor(state_dict, [f"{prefix}.ssm.xProj.weight", f"{prefix}.ssm.x_proj.weight"]).detach().cpu().numpy().astype(np.float32),
            "dtProjWeight": get_first_tensor(state_dict, [f"{prefix}.ssm.dtProj.weight", f"{prefix}.ssm.dt_proj.weight"]).detach().cpu().numpy().astype(np.float32),
            "dtProjBias": get_first_tensor(state_dict, [f"{prefix}.ssm.dtProj.bias", f"{prefix}.ssm.dt_proj.bias"]).detach().cpu().numpy().astype(np.float32),
            "outProj": get_first_tensor(state_dict, [f"{prefix}.ssm.outProj.weight", f"{prefix}.ssm.out_proj.weight"]).detach().cpu().numpy().astype(np.float32),
        }
        layer_arrays["d_inner"] = np.array(model.dModel * model.expand, dtype=np.int32)
        layer_arrays["d_state"] = np.array(model.dState, dtype=np.int32)
        layer_arrays["d_conv"] = np.array(model.dConv, dtype=np.int32)
        layer_arrays["dt_rank"] = np.array(model.dtRank, dtype=np.int32)
        layer_arrays["scanImpl"] = scan_impl
        layers.append(layer_arrays)

    if model.gatedAttention is None:
        raise ValueError("CI-BabyMamba export expects gated attention to be enabled.")

    return {
        "config": {
            "variant": "ciBabyMambaHar",
            "scan_impl": scan_impl,
            "seq_len": model.seqLen,
            "in_channels": model.inChannels,
            "num_classes": model.numClasses,
            "d_model": model.dModel,
            "d_state": model.dState,
            "expand": model.expand,
            "d_inner": model.dModel * model.expand,
            "dt_rank": model.dtRank,
            "d_conv": model.dConv,
            "n_layers": model.nLayers,
            "stem_kernel": model.stemKernel,
            "stem_padding": model.stemKernel // 2,
            "patch_kernel": model.patchKernel,
            "patch_stride": model.patchStride,
            "patch_padding": model.patchKernel // 4,
            "patch_out_len": patch_out_len,
        },
        "stemWeight": stem_weight,
        "stemBias": stem_bias,
        "patchDepthwise": state_dict["patchDepthwise.weight"].squeeze(1).detach().cpu().numpy().astype(np.float32),
        "patchPointwise": patch_pointwise,
        "patchBias": patch_bias,
        "posEmbed": pos_embed,
        "layers": layers,
        "gatedProjectionWeight": state_dict["gatedAttention.projection.weight"].detach().cpu().numpy().astype(np.float32),
        "gatedProjectionBias": state_dict["gatedAttention.projection.bias"].detach().cpu().numpy().astype(np.float32),
        "gatedContext": state_dict["gatedAttention.context"].detach().cpu().numpy().astype(np.float32),
        "headNormWeight": state_dict["headNorm.weight"].detach().cpu().numpy().astype(np.float32),
        "headNormBias": state_dict["headNorm.bias"].detach().cpu().numpy().astype(np.float32),
        "headLinearWeight": state_dict["headLinear.weight"].detach().cpu().numpy().astype(np.float32),
        "headLinearBias": state_dict["headLinear.bias"].detach().cpu().numpy().astype(np.float32),
    }


def cpp_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return f"{text}f"


def format_1d(name: str, array: np.ndarray) -> str:
    values = ", ".join(cpp_float(v) for v in array.reshape(-1))
    return f"static const float {name}[{array.size}] = {{{values}}};\n"


def format_2d(name: str, array: np.ndarray) -> str:
    rows, cols = array.shape
    lines = [f"static const float {name}[{rows}][{cols}] = {{"]
    for row in array:
        lines.append("  {" + ", ".join(cpp_float(v) for v in row) + "},")
    lines.append("};\n")
    return "\n".join(lines)


def format_3d(name: str, array: np.ndarray) -> str:
    d0, d1, d2 = array.shape
    lines = [f"static const float {name}[{d0}][{d1}][{d2}] = {{"]
    for plane in array:
        lines.append("  {")
        for row in plane:
            lines.append("    {" + ", ".join(cpp_float(v) for v in row) + "},")
        lines.append("  },")
    lines.append("};\n")
    return "\n".join(lines)


def write_crossover_header(
    export_dir: Path,
    spec: DatasetSpec,
    seed: int,
    arrays: Dict[str, object],
    sample: np.ndarray,
    label: int,
    pytorch_logits: np.ndarray,
    engine_logits: np.ndarray,
) -> None:
    config = arrays["config"]
    assert isinstance(config, dict)
    lines: List[str] = [
        "#pragma once",
        "",
        f"#define BABYMAMBA_VARIANT_CROSSOVER 1",
        f"#define BABYMAMBA_VARIANT_CHANNEL_INDEPENDENT 0",
        f"#define BABYMAMBA_DATASET_NAME \"{spec.key}\"",
        f"#define BABYMAMBA_DISPLAY_NAME \"{spec.display_name}\"",
        f"#define BABYMAMBA_SEED {seed}",
        f"#define BABYMAMBA_SEQ_LEN {config['seq_len']}",
        f"#define BABYMAMBA_IN_CHANNELS {config['in_channels']}",
        f"#define BABYMAMBA_NUM_CLASSES {config['num_classes']}",
        f"#define BABYMAMBA_D_MODEL {config['d_model']}",
        f"#define BABYMAMBA_D_STATE {config['d_state']}",
        f"#define BABYMAMBA_D_INNER {config['d_inner']}",
        f"#define BABYMAMBA_DT_RANK {config['dt_rank']}",
        f"#define BABYMAMBA_D_CONV {config['d_conv']}",
        f"#define BABYMAMBA_NUM_LAYERS {config['n_layers']}",
        f"#define BABYMAMBA_STEM_KERNEL {config['stem_kernel']}",
        f"#define BABYMAMBA_PATCH_KERNEL {config['patch_kernel']}",
        f"#define BABYMAMBA_PATCH_STRIDE {config['patch_stride']}",
        f"#define BABYMAMBA_PATCH_OUT_LEN {config['patch_out_len']}",
        f"#define BABYMAMBA_SCAN_IMPL_FALLBACK {1 if config.get('scan_impl') == 'fallback_chunked' else 0}",
        f"#define BABYMAMBA_FIXTURE_LABEL {label}",
        "",
    ]

    class_names = ", ".join(f"\"{name}\"" for name in spec.class_names)
    lines.append(f"static const char* const kBabyMambaClassNames[BABYMAMBA_NUM_CLASSES] = {{{class_names}}};")
    lines.append("")

    lines.append(format_3d("kStemWeight", np.asarray(arrays["stemWeight"], dtype=np.float32)))
    lines.append(format_1d("kStemBias", np.asarray(arrays["stemBias"], dtype=np.float32)))
    lines.append(format_2d("kPatchDepthwise", np.asarray(arrays["patchDepthwise"], dtype=np.float32)))
    lines.append(format_2d("kPatchPointwise", np.asarray(arrays["patchPointwise"], dtype=np.float32)))
    lines.append(format_1d("kPatchBias", np.asarray(arrays["patchBias"], dtype=np.float32)))
    lines.append(format_3d("kPosEmbed", np.asarray(arrays["posEmbed"], dtype=np.float32)))

    layers = arrays["layers"]
    assert isinstance(layers, list)
    for idx, layer in enumerate(layers):
        assert isinstance(layer, dict)
        prefix = f"kLayer{idx}"
        lines.append(format_1d(f"{prefix}PreNormWeight", np.asarray(layer["preNormWeight"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}PreNormBias", np.asarray(layer["preNormBias"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}PostNormWeight", np.asarray(layer["postNormWeight"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}PostNormBias", np.asarray(layer["postNormBias"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}ALog", np.asarray(layer["ALog"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}D", np.asarray(layer["D"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}InProj", np.asarray(layer["inProj"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}Conv1dWeight", np.asarray(layer["conv1dWeight"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}Conv1dBias", np.asarray(layer["conv1dBias"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}XProj", np.asarray(layer["xProj"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}DtProjWeight", np.asarray(layer["dtProjWeight"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}DtProjBias", np.asarray(layer["dtProjBias"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}OutProj", np.asarray(layer["outProj"], dtype=np.float32)))

    lines.append(format_1d("kHeadNormWeight", np.asarray(arrays["headNormWeight"], dtype=np.float32)))
    lines.append(format_1d("kHeadNormBias", np.asarray(arrays["headNormBias"], dtype=np.float32)))
    lines.append(format_2d("kHeadLinearWeight", np.asarray(arrays["headLinearWeight"], dtype=np.float32)))
    lines.append(format_1d("kHeadLinearBias", np.asarray(arrays["headLinearBias"], dtype=np.float32)))

    flat_input = sample.reshape(-1).astype(np.float32)
    lines.append(f"static const float kFixtureInput[{flat_input.size}] = {{{', '.join(cpp_float(v) for v in flat_input)}}};")
    lines.append(f"static const float kFixturePyTorchLogits[{pytorch_logits.size}] = {{{', '.join(cpp_float(v) for v in pytorch_logits)}}};")
    lines.append(f"static const float kFixtureEngineLogits[{engine_logits.size}] = {{{', '.join(cpp_float(v) for v in engine_logits)}}};")
    lines.append("")

    (export_dir / "babyMambaWeights.h").write_text("\n".join(lines), encoding="ascii")


def write_ci_header(
    export_dir: Path,
    spec: DatasetSpec,
    seed: int,
    arrays: Dict[str, object],
    sample: np.ndarray,
    label: int,
    pytorch_logits: np.ndarray,
    engine_logits: np.ndarray,
) -> None:
    config = arrays["config"]
    assert isinstance(config, dict)
    lines: List[str] = [
        "#pragma once",
        "",
        f"#define BABYMAMBA_VARIANT_CROSSOVER 0",
        f"#define BABYMAMBA_VARIANT_CHANNEL_INDEPENDENT 1",
        f"#define BABYMAMBA_DATASET_NAME \"{spec.key}\"",
        f"#define BABYMAMBA_DISPLAY_NAME \"{spec.display_name}\"",
        f"#define BABYMAMBA_SEED {seed}",
        f"#define BABYMAMBA_SEQ_LEN {config['seq_len']}",
        f"#define BABYMAMBA_IN_CHANNELS {config['in_channels']}",
        f"#define BABYMAMBA_NUM_CLASSES {config['num_classes']}",
        f"#define BABYMAMBA_D_MODEL {config['d_model']}",
        f"#define BABYMAMBA_D_STATE {config['d_state']}",
        f"#define BABYMAMBA_D_INNER {config['d_inner']}",
        f"#define BABYMAMBA_DT_RANK {config['dt_rank']}",
        f"#define BABYMAMBA_D_CONV {config['d_conv']}",
        f"#define BABYMAMBA_NUM_LAYERS {config['n_layers']}",
        f"#define BABYMAMBA_STEM_KERNEL {config['stem_kernel']}",
        f"#define BABYMAMBA_PATCH_KERNEL {config['patch_kernel']}",
        f"#define BABYMAMBA_PATCH_STRIDE {config['patch_stride']}",
        f"#define BABYMAMBA_PATCH_OUT_LEN {config['patch_out_len']}",
        f"#define BABYMAMBA_SCAN_IMPL_FALLBACK {1 if config.get('scan_impl') == 'fallback_chunked' else 0}",
        f"#define BABYMAMBA_FIXTURE_LABEL {label}",
        "",
    ]

    class_names = ", ".join(f"\"{name}\"" for name in spec.class_names)
    lines.append(f"static const char* const kBabyMambaClassNames[BABYMAMBA_NUM_CLASSES] = {{{class_names}}};")
    lines.append("")

    lines.append(format_3d("kStemWeight", np.asarray(arrays["stemWeight"], dtype=np.float32)))
    lines.append(format_1d("kStemBias", np.asarray(arrays["stemBias"], dtype=np.float32)))
    lines.append(format_2d("kPatchDepthwise", np.asarray(arrays["patchDepthwise"], dtype=np.float32)))
    lines.append(format_2d("kPatchPointwise", np.asarray(arrays["patchPointwise"], dtype=np.float32)))
    lines.append(format_1d("kPatchBias", np.asarray(arrays["patchBias"], dtype=np.float32)))
    lines.append(format_3d("kPosEmbed", np.asarray(arrays["posEmbed"], dtype=np.float32)))

    layers = arrays["layers"]
    assert isinstance(layers, list)
    for idx, layer in enumerate(layers):
        assert isinstance(layer, dict)
        prefix = f"kLayer{idx}"
        lines.append(format_1d(f"{prefix}PreNormWeight", np.asarray(layer["preNormWeight"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}PreNormBias", np.asarray(layer["preNormBias"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}PostNormWeight", np.asarray(layer["postNormWeight"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}PostNormBias", np.asarray(layer["postNormBias"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}ALog", np.asarray(layer["ALog"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}D", np.asarray(layer["D"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}InProj", np.asarray(layer["inProj"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}Conv1dWeight", np.asarray(layer["conv1dWeight"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}Conv1dBias", np.asarray(layer["conv1dBias"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}XProj", np.asarray(layer["xProj"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}DtProjWeight", np.asarray(layer["dtProjWeight"], dtype=np.float32)))
        lines.append(format_1d(f"{prefix}DtProjBias", np.asarray(layer["dtProjBias"], dtype=np.float32)))
        lines.append(format_2d(f"{prefix}OutProj", np.asarray(layer["outProj"], dtype=np.float32)))

    lines.append(format_2d("kGatedProjectionWeight", np.asarray(arrays["gatedProjectionWeight"], dtype=np.float32)))
    lines.append(format_1d("kGatedProjectionBias", np.asarray(arrays["gatedProjectionBias"], dtype=np.float32)))
    lines.append(format_1d("kGatedContext", np.asarray(arrays["gatedContext"], dtype=np.float32)))
    lines.append(format_1d("kHeadNormWeight", np.asarray(arrays["headNormWeight"], dtype=np.float32)))
    lines.append(format_1d("kHeadNormBias", np.asarray(arrays["headNormBias"], dtype=np.float32)))
    lines.append(format_2d("kHeadLinearWeight", np.asarray(arrays["headLinearWeight"], dtype=np.float32)))
    lines.append(format_1d("kHeadLinearBias", np.asarray(arrays["headLinearBias"], dtype=np.float32)))

    flat_input = sample.reshape(-1).astype(np.float32)
    lines.append(f"static const float kFixtureInput[{flat_input.size}] = {{{', '.join(cpp_float(v) for v in flat_input)}}};")
    lines.append(f"static const float kFixturePyTorchLogits[{pytorch_logits.size}] = {{{', '.join(cpp_float(v) for v in pytorch_logits)}}};")
    lines.append(f"static const float kFixtureEngineLogits[{engine_logits.size}] = {{{', '.join(cpp_float(v) for v in engine_logits)}}};")
    lines.append("")

    (export_dir / "babyMambaWeights.h").write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    dataset_keys = parse_csv_or_all(args.datasets, tuple(DATASET_SPECS.keys()))

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_payload: Dict[str, object] = {
        "variant": args.variant,
        "requested_datasets": dataset_keys,
        "exports": [],
        "missing": [],
    }

    for dataset_key in dataset_keys:
        spec = DATASET_SPECS[dataset_key]
        try:
            if args.variant == "crossoverBiDirBabyMambaHar":
                checkpoint_path, seed, run_dir = find_checkpoint_for_crossover(dataset_key, args.seed)
            else:
                checkpoint_path, seed, run_dir = find_checkpoint_for_ci(dataset_key, args.seed)
        except Exception as exc:
            summary_payload["missing"].append({"dataset": dataset_key, "reason": str(exc)})
            if args.strict:
                raise
            print(f"[missing] {dataset_key}: {exc}")
            continue

        export_dir = args.output_root / args.variant / dataset_key
        export_dir.mkdir(parents=True, exist_ok=True)

        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = extract_state_dict(payload)
        uses_fallback_style = state_dict_uses_fallback_style(state_dict)
        model = build_model(args.variant, dataset_key, dropout=0.0, force_fallback=uses_fallback_style)
        model.load_state_dict(state_dict, strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        samples = load_dataset_test_samples(dataset_key)
        if args.sample_index >= len(samples):
            raise IndexError(f"Sample index {args.sample_index} out of range for {dataset_key}")
        sample, label = samples[args.sample_index]
        sample_batch = torch.from_numpy(sample[np.newaxis, ...]).to(device)
        with torch.no_grad():
            pytorch_logits = model(sample_batch).detach().cpu().numpy().astype(np.float32).reshape(-1)

        if args.variant == "crossoverBiDirBabyMambaHar":
            arrays = numpy_arrays_for_crossover(model)
            engine_logits = run_crossover_reference(arrays, sample)
        else:
            arrays = numpy_arrays_for_ci(model)
            engine_logits = run_ci_reference(arrays, sample)

        parity = 100.0 * (1.0 - float(np.mean(np.abs(engine_logits - pytorch_logits)) / (np.mean(np.abs(pytorch_logits)) + 1e-6)))
        manifest = {
            "variant": args.variant,
            "dataset": dataset_key,
            "checkpoint_path": str(checkpoint_path),
            "run_dir": str(run_dir),
            "seed": seed,
            "sample_index": args.sample_index,
            "label": label,
            "pytorch_top1": int(np.argmax(pytorch_logits)),
            "engine_top1": int(np.argmax(engine_logits)),
            "parity_percent": parity,
            "config": arrays["config"],
            "class_names": list(spec.class_names),
        }

        if args.variant == "crossoverBiDirBabyMambaHar":
            write_crossover_header(
                export_dir=export_dir,
                spec=spec,
                seed=seed,
                arrays=arrays,
                sample=sample,
                label=label,
                pytorch_logits=pytorch_logits,
                engine_logits=engine_logits,
            )
        else:
            write_ci_header(
                export_dir=export_dir,
                spec=spec,
                seed=seed,
                arrays=arrays,
                sample=sample,
                label=label,
                pytorch_logits=pytorch_logits,
                engine_logits=engine_logits,
            )
        (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        summary_payload["exports"].append(manifest)
        print(f"[exported] {args.variant} {dataset_key} seed={seed} parity={parity:.4f}%")

    (args.output_root / f"{args.variant}_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

