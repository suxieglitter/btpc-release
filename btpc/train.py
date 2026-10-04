"""Stage 1 / Stage 2 training pipeline with anchor-free periodic filtering.

Stage 1 trains a Barlow Twins encoder on two augmented views of each P-wave
window and periodically scores every sample with an unsupervised reliability
score (waveform quality, augmentation consistency, neighbor agreement, epoch
stability, cluster margin). Low-scoring samples are down-weighted or moved to
a reject pool. Stage 2 clusters the encoder features into two pseudo-classes
and trains a classification head on the retained samples.

Command line entry point: ``btpc-train`` (subcommands ``stage1``, ``stage2``,
``eval``). Hyperparameter defaults reproduce the reference run documented in
``configs/btpc_0518.yaml``.
"""

import argparse
import csv
import glob
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from .data import (
    DEFAULT_LABEL_KEY,
    DEFAULT_SNR_KEY,
    DEFAULT_WAVEFORM_KEY,
    PseudoLabelDataset,
    SeismicPolarityDataset,
    build_eval_dataloader,
    build_train_dataloader,
    load_ridgecrest_unlabeled_dataset,
    load_scsn_polarity_dataset,
    weak_eval_augment_waveforms,
)
from .models import build_barlow_model, build_stage2_classifier
from .utils import (
    acc,
    assign_binary_clusters_by_centers,
    checked_path,
    cluster,
    compute_binary_cluster_centers,
    compute_cluster_margin,
    csv_text,
    dedim,
    device,
    load_checkpoint,
    maybe_mkdir,
    normalize_percentile,
    preprocess_cluster_features,
    print_score_summary,
    resolve_data_path,
    save_confusion_matrix_percent,
    save_json,
    summarize_score_array,
)

STATUS_RETAIN = "retain"
STATUS_DOWNWEIGHT = "downweight"
STATUS_REJECT_CANDIDATE = "reject_candidate"
STATUS_REJECT_POOL = "reject_pool"
STATUS_TO_CODE = {
    STATUS_RETAIN: 0,
    STATUS_DOWNWEIGHT: 1,
    STATUS_REJECT_CANDIDATE: 2,
    STATUS_REJECT_POOL: 3,
}
CODE_TO_STATUS = {value: key for key, value in STATUS_TO_CODE.items()}

# Reference hyperparameters (mdl_d20260518 run "..._encoder_max_clean_42").
# Only ``data_path`` has no default: data never ships with the package.
STAGE1_DEFAULTS = {
    "data_path": None,
    "dataset_source": "scsn",
    "waveform_key": DEFAULT_WAVEFORM_KEY,
    "label_key": DEFAULT_LABEL_KEY,
    "snr_key": DEFAULT_SNR_KEY,
    "snr_min": 0.0,
    "snr_max": 1000.0,
    "num_used": 10000,
    "resize": 32,
    "shift": 0,
    "bino": True,
    "base_cha": 16,
    "projector_dims": 512,
    "batch_size": 256,
    "lr": 1e-3,
    "lambda_param": 0.005,
    "epochs": 80,
    "cluster_feature": "encoder",
    "norm_mod": "max",
    "eval_source": "clean",
    "aug_shift": 1,
    "aug_noise_min": 0.05,
    "aug_noise_max": 0.2,
    "aug_scale_min": 0.8,
    "aug_scale_max": 1.2,
    "enable_periodic_filtering": True,
    "stage1_cluster_method": "spectral",
    "warmup_epochs": 20,
    "filter_interval": 10,
    "eval_interval": 5,
    "n_tta_views": 30,
    "tta_max_shift": 2,
    "knn_k": 10,
    "low_score_threshold": 0.45,
    "high_score_threshold": 0.7,
    "low_score_patience": 2,
    "w_q": 0.15,
    "w_aug": 0.2,
    "w_knn": 0.2,
    "w_stab": 0.25,
    "w_margin": 0.2,
    "downweight_loss_scale": 0.35,
    "stability_ema_alpha": 0.5,
    "seed": 42,
}

STAGE2_DEFAULTS = {
    "stage1_dir": None,
    "stage2_save_dir": None,
    "stage2_cluster_method": "spectral",
    "stage2_epochs": 30,
    "stage2_lr": 1e-3,
    "stage2_weight_decay": 1e-4,
    "stage2_include_downweight": True,
    "whether_freeze_encoder_in_stage2": True,
    "stage2_head_hidden_dim": 0,
    "batch_size": None,
}


def get_data_selection_seed(config: Dict) -> int:
    return int(config.get("data_selection_seed", config.get("seed", 42)))


def build_default_save_path(cfg: Dict) -> str:
    """Descriptive run-directory name, matching the historical layout."""
    snr_lo, snr_hi = cfg["snr_range"]
    noise_lo, noise_hi = cfg["aug_noise_std_range"]
    scale_lo, scale_hi = cfg["aug_scale_range"]
    return (
        f"{cfg['aug_shift']}_{noise_lo}-{noise_hi}"
        f"_{scale_lo}-{scale_hi}_{cfg['lambda_param']}"
        f"_{snr_lo}-{snr_hi}_{cfg['num_used']}_{cfg['resize']}_{cfg['shift']}"
        f"_{cfg['bino']}_{cfg['base_cha']}_{cfg['projector_dims']}_{cfg['batch_size']}"
        f"_{cfg['lr']}_{cfg['epochs']}"
        f"_{cfg['cluster_feature']}_{cfg['norm_mod']}_{cfg['eval_source']}_{cfg['seed']}"
    )


def load_yaml_config(path: str, allowed_keys) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("--config requires PyYAML. Install it with: pip install pyyaml") from exc
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    unknown = sorted(set(loaded) - set(allowed_keys))
    if unknown:
        raise ValueError(f"Unknown config key(s) in {path}: {unknown}")
    return loaded


def resolve_stage1_config(args) -> Dict:
    """Merge built-in defaults <- YAML config <- CLI flags (in that order)."""
    cfg = dict(STAGE1_DEFAULTS)
    if getattr(args, "config", None):
        allowed = set(STAGE1_DEFAULTS) | {"data_selection_seed"}
        cfg.update(load_yaml_config(args.config, allowed))
    for key, value in vars(args).items():
        if key in cfg and value is not None:
            cfg[key] = value
    cfg["snr_range"] = [float(cfg.pop("snr_min")), float(cfg.pop("snr_max"))]
    cfg["aug_noise_std_range"] = [float(cfg.pop("aug_noise_min")), float(cfg.pop("aug_noise_max"))]
    cfg["aug_scale_range"] = [float(cfg.pop("aug_scale_min")), float(cfg.pop("aug_scale_max"))]
    cfg["data_selection_seed"] = int(cfg.get("data_selection_seed", cfg["seed"]))
    return cfg


def resolve_stage2_config(args) -> Dict:
    cfg = dict(STAGE2_DEFAULTS)
    for key, value in vars(args).items():
        if key in cfg and value is not None:
            cfg[key] = value
    return cfg


def stage1_runtime_config(cfg: Dict, data_path: str) -> Dict:
    """Config dict saved as stage1_config.json (keys match released checkpoints)."""
    return {
        "train_data_path": data_path,
        "dataset_source": str(cfg["dataset_source"]),
        "data_waveform_key": cfg["waveform_key"],
        "data_label_key": cfg["label_key"],
        "data_snr_key": cfg["snr_key"],
        "snr_range": [float(v) for v in cfg["snr_range"]],
        "num_used": int(cfg["num_used"]),
        "resize": int(cfg["resize"]),
        "shift": int(cfg["shift"]),
        "bino": bool(cfg["bino"]),
        "base_cha": int(cfg["base_cha"]),
        "projector_dims": cfg["projector_dims"],
        "batch_size": int(cfg["batch_size"]),
        "lr": float(cfg["lr"]),
        "lambda_param": float(cfg["lambda_param"]),
        "epochs": int(cfg["epochs"]),
        "cluster_feature": str(cfg["cluster_feature"]),
        "norm_mod": str(cfg["norm_mod"]),
        "eval_source": str(cfg["eval_source"]),
        "aug_shift": int(cfg["aug_shift"]),
        "aug_noise_std_range": [float(v) for v in cfg["aug_noise_std_range"]],
        "aug_scale_range": [float(v) for v in cfg["aug_scale_range"]],
        "enable_periodic_filtering": bool(cfg["enable_periodic_filtering"]),
        "stage1_cluster_method": str(cfg["stage1_cluster_method"]),
        "warmup_epochs": int(cfg["warmup_epochs"]),
        "filter_interval": int(cfg["filter_interval"]),
        "n_tta_views": int(cfg["n_tta_views"]),
        "tta_max_shift": int(cfg["tta_max_shift"]),
        "knn_k": int(cfg["knn_k"]),
        "low_score_threshold": float(cfg["low_score_threshold"]),
        "high_score_threshold": float(cfg["high_score_threshold"]),
        "low_score_patience": int(cfg["low_score_patience"]),
        "w_q": float(cfg["w_q"]),
        "w_aug": float(cfg["w_aug"]),
        "w_knn": float(cfg["w_knn"]),
        "w_stab": float(cfg["w_stab"]),
        "w_margin": float(cfg["w_margin"]),
        "downweight_loss_scale": float(cfg["downweight_loss_scale"]),
        "stability_ema_alpha": float(cfg["stability_ema_alpha"]),
        "seed": int(cfg["seed"]),
        "data_selection_seed": int(cfg["data_selection_seed"]),
    }


def stage1_data_path_from_config(stage1_config: Dict) -> str:
    data_path = stage1_config.get("train_data_path") or stage1_config.get("data_path")
    if not data_path:
        raise ValueError(
            "stage1_config.json does not record a training data path; "
            "the training HDF5 file is required for this step."
        )
    return resolve_data_path(data_path)


def dataset_from_stage1_config(stage1_config: Dict) -> SeismicPolarityDataset:
    """Rebuild the Stage 1 training dataset from a saved stage1_config.json."""
    common_kwargs = dict(
        data_path=stage1_data_path_from_config(stage1_config),
        snr_range=stage1_config["snr_range"],
        n_select=stage1_config["num_used"],
        resize=stage1_config["resize"],
        shift=stage1_config["shift"],
        aug_shift=stage1_config["aug_shift"],
        aug_noise_std_range=stage1_config["aug_noise_std_range"],
        aug_scale_range=stage1_config["aug_scale_range"],
        norm_mod=stage1_config["norm_mod"],
    )
    if str(stage1_config.get("dataset_source", "scsn")) == "ridgecrest_unlabeled":
        return load_ridgecrest_unlabeled_dataset(**common_kwargs)
    return load_scsn_polarity_dataset(
        bino=stage1_config["bino"],
        waveform_key=stage1_config.get("data_waveform_key", DEFAULT_WAVEFORM_KEY),
        label_key=stage1_config.get("data_label_key", DEFAULT_LABEL_KEY),
        snr_key=stage1_config.get("data_snr_key", DEFAULT_SNR_KEY),
        selection_seed=get_data_selection_seed(stage1_config),
        **common_kwargs,
    )


# ==================== Unsupervised uncertainty scores ====================


def compute_neighbor_agreement(embeddings, cluster_assignments, k=10, metric="cosine"):
    """Fraction of each sample's k nearest neighbors sharing its cluster label."""
    emb = np.asarray(embeddings, dtype=float)
    assignments = np.asarray(cluster_assignments, dtype=int)
    if emb.ndim != 2 or assignments.ndim != 1 or emb.shape[0] != assignments.shape[0]:
        raise ValueError(
            "embeddings (n_samples, n_features) and cluster_assignments (n_samples,) "
            "must be aligned 1-D/2-D arrays."
        )
    if emb.shape[0] < 2:
        raise ValueError("At least two samples are required to compute neighbor agreement.")
    if k <= 0:
        raise ValueError(f"k must be a positive integer; got {k}.")

    effective_k = min(int(k), emb.shape[0] - 1)
    neighbors = NearestNeighbors(n_neighbors=effective_k + 1, metric=metric)
    neighbors.fit(emb)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    agreement = assignments[indices] == assignments[:, None]
    return np.clip(agreement.mean(axis=1), 0.0, 1.0)


def compute_augmentation_consistency(repeated_assignments, metric="label"):
    """Fraction of repeated augmented views agreeing with the modal cluster ID.

    ``repeated_assignments`` has shape ``(n_repeats, n_samples)`` (a
    ``(n_samples, n_repeats)`` input is transposed automatically).
    """
    if str(metric).lower() != "label":
        raise ValueError(f"Unsupported metric '{metric}'; only 'label' is implemented.")
    assignments = np.asarray(repeated_assignments, dtype=int)
    if assignments.ndim != 2 or 0 in assignments.shape:
        raise ValueError(
            "repeated_assignments must have shape (n_repeats, n_samples) or "
            "(n_samples, n_repeats)."
        )
    if assignments.shape[0] > assignments.shape[1]:
        assignments = assignments.T
    n_repeats, n_samples = assignments.shape
    if n_repeats == 1:
        return np.ones(n_samples, dtype=float)

    scores = np.empty(n_samples, dtype=float)
    for sample_idx in range(n_samples):
        counts = np.unique(assignments[:, sample_idx], return_counts=True)[1]
        scores[sample_idx] = counts.max() / float(n_repeats)
    return np.clip(scores, 0.0, 1.0)


def compute_waveform_quality_scores(
    dataset: SeismicPolarityDataset,
    indices: np.ndarray,
):
    # Reuse the dataset-provided SNR directly. In this project it is already
    # defined as max|post 0.5s| / max|pre 0.5s|, so we should not recompute it.
    raw_scores = np.asarray(dataset.snrs[np.asarray(indices, dtype=int)], dtype=float)
    return raw_scores, normalize_percentile(raw_scores)


def compute_filter_augmentation_consistency(
    model,
    clean_waveforms: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    cluster_centers: np.ndarray,
    views: int,
    max_shift: int,
    batch_size_local: int,
    cluster_feature: str = "encoder",
) -> np.ndarray:
    if views <= 1:
        return np.ones(clean_waveforms.shape[0], dtype=float)

    repeated_labels = []
    for _ in range(views):
        aug_waveforms = weak_eval_augment_waveforms(
            clean_waveforms,
            max_shift=max_shift,
            scale_jitter=0.02,
            noise_std=0.005,
        )
        aug_raw = encode_waveforms(
            model, aug_waveforms, batch_size=batch_size_local, cluster_feature=cluster_feature
        )
        aug_feat, _, _ = preprocess_cluster_features(
            aug_raw,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )
        aug_labels, _ = assign_binary_clusters_by_centers(aug_feat, cluster_centers)
        repeated_labels.append(aug_labels)
    repeated_labels = np.stack(repeated_labels, axis=0)
    return compute_augmentation_consistency(
        repeated_assignments=repeated_labels, metric="label"
    )


def align_binary_clusters_with_previous(
    prev_labels_full: Optional[np.ndarray],
    current_indices: np.ndarray,
    current_labels: np.ndarray,
    current_centers: np.ndarray,
):
    """Flip the current 0/1 labeling if it swaps relative to the previous one."""
    current_indices = np.asarray(current_indices, dtype=int)
    current_labels = np.asarray(current_labels, dtype=int)
    aligned_labels = current_labels.copy()
    aligned_centers = current_centers.copy()
    instant_stability = np.ones(len(current_indices), dtype=float)

    if prev_labels_full is None:
        return aligned_labels, aligned_centers, instant_stability

    prev_shared = np.asarray(prev_labels_full[current_indices], dtype=int)
    shared_mask = prev_shared >= 0
    if not np.any(shared_mask):
        instant_stability[:] = 0.5
        return aligned_labels, aligned_centers, instant_stability

    overlap = np.zeros((2, 2), dtype=int)
    for prev_label, curr_label in zip(prev_shared[shared_mask], current_labels[shared_mask]):
        if prev_label in (0, 1) and curr_label in (0, 1):
            overlap[prev_label, curr_label] += 1

    same = overlap[0, 0] + overlap[1, 1]
    swap = overlap[0, 1] + overlap[1, 0]
    if swap > same:
        aligned_labels = 1 - current_labels
        aligned_centers = current_centers[[1, 0]]

    instant_stability[:] = 0.5
    instant_stability[shared_mask] = (
        aligned_labels[shared_mask] == prev_shared[shared_mask]
    ).astype(float)
    return aligned_labels, aligned_centers, instant_stability


def update_filter_membership(
    reliability_score: np.ndarray,
    previous_low_counts: Optional[np.ndarray],
    previous_reject_pool: Optional[np.ndarray],
    low_threshold: float,
    high_threshold: float,
    patience: int,
    downweight_scale: float,
):
    score = np.asarray(reliability_score, dtype=float)
    low_counts = (
        np.zeros_like(score, dtype=np.int64)
        if previous_low_counts is None
        else previous_low_counts.copy()
    )
    reject_pool = (
        np.zeros_like(score, dtype=bool)
        if previous_reject_pool is None
        else np.asarray(previous_reject_pool, dtype=bool).copy()
    )

    status = np.full(score.shape, STATUS_RETAIN, dtype=object)
    sample_weights = np.ones(score.shape, dtype=float)

    for idx in range(score.shape[0]):
        if reject_pool[idx]:
            status[idx] = STATUS_REJECT_POOL
            sample_weights[idx] = 0.0
            continue

        if score[idx] >= high_threshold:
            status[idx] = STATUS_RETAIN
            low_counts[idx] = 0
            sample_weights[idx] = 1.0
        elif score[idx] >= low_threshold:
            status[idx] = STATUS_DOWNWEIGHT
            low_counts[idx] = 0
            sample_weights[idx] = float(downweight_scale)
        else:
            low_counts[idx] += 1
            if low_counts[idx] >= patience:
                reject_pool[idx] = True
                status[idx] = STATUS_REJECT_POOL
                sample_weights[idx] = 0.0
            else:
                status[idx] = STATUS_REJECT_CANDIDATE
                sample_weights[idx] = float(downweight_scale)

    return status, low_counts, sample_weights, reject_pool


def fuse_reliability_scores(
    waveform_quality: np.ndarray,
    augmentation_consistency: np.ndarray,
    neighbor_agreement: np.ndarray,
    epoch_stability_values: np.ndarray,
    cluster_margin_values: np.ndarray,
    weight_dict: Dict[str, float],
) -> np.ndarray:
    weights = np.array(
        [
            float(weight_dict["w_q"]),
            float(weight_dict["w_aug"]),
            float(weight_dict["w_knn"]),
            float(weight_dict["w_stab"]),
            float(weight_dict["w_margin"]),
        ],
        dtype=float,
    )
    weights = weights / max(weights.sum(), 1e-12)
    stacked = np.vstack(
        [
            np.clip(waveform_quality, 0.0, 1.0),
            np.clip(augmentation_consistency, 0.0, 1.0),
            np.clip(neighbor_agreement, 0.0, 1.0),
            np.clip(epoch_stability_values, 0.0, 1.0),
            np.clip(cluster_margin_values, 0.0, 1.0),
        ]
    )
    return np.clip(weights @ stacked, 0.0, 1.0)


def init_filter_state(n_samples: int) -> Dict:
    return {
        "epoch": 0,
        "reliability_score": np.ones(n_samples, dtype=float),
        "waveform_quality_raw": np.ones(n_samples, dtype=float),
        "waveform_quality": np.ones(n_samples, dtype=float),
        "augmentation_consistency": np.ones(n_samples, dtype=float),
        "neighbor_agreement": np.ones(n_samples, dtype=float),
        "epoch_stability": np.ones(n_samples, dtype=float),
        "cluster_margin": np.ones(n_samples, dtype=float),
        "aligned_cluster_labels": np.full(n_samples, -1, dtype=np.int64),
        "status": np.full(n_samples, STATUS_RETAIN, dtype=object),
        "status_codes": np.full(n_samples, STATUS_TO_CODE[STATUS_RETAIN], dtype=np.int64),
        "low_score_count": np.zeros(n_samples, dtype=np.int64),
        "sample_weight": np.ones(n_samples, dtype=float),
        "reject_pool_mask": np.zeros(n_samples, dtype=bool),
        "cluster_centers": np.empty((0, 0), dtype=float),
        "feature_mean": np.empty((0,), dtype=float),
        "feature_scale": np.empty((0,), dtype=float),
        "filter_indices": np.arange(n_samples, dtype=np.int64),
    }


def parse_epoch_from_filter_csv(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.split("_")[-1])


def export_filtering_timeline(filter_dir: str, low_threshold: float, high_threshold: float):
    csv_paths = sorted(
        glob.glob(os.path.join(filter_dir, "filter_epoch_*.csv")),
        key=parse_epoch_from_filter_csv,
    )
    if not csv_paths:
        return

    metric_names = [
        "reliability_score",
        "waveform_quality",
        "augmentation_consistency",
        "neighbor_agreement",
        "epoch_stability",
        "cluster_margin",
    ]
    timeline_rows = []
    for csv_path in csv_paths:
        epoch = parse_epoch_from_filter_csv(csv_path)
        counts = {
            STATUS_RETAIN: 0,
            STATUS_DOWNWEIGHT: 0,
            STATUS_REJECT_CANDIDATE: 0,
            STATUS_REJECT_POOL: 0,
        }
        metric_values = {name: [] for name in metric_names}

        text = Path(csv_path).read_text(encoding="utf-8")
        for row in csv.DictReader(io.StringIO(text)):
            status_name = str(row["status"])
            if status_name in counts:
                counts[status_name] += 1
            for metric_name in metric_names:
                metric_values[metric_name].append(float(row[metric_name]))

        total_count = int(sum(counts.values()))
        row = {
            "epoch": epoch,
            "total_count": total_count,
            "retain_count": counts[STATUS_RETAIN],
            "downweight_count": counts[STATUS_DOWNWEIGHT],
            "reject_candidate_count": counts[STATUS_REJECT_CANDIDATE],
            "reject_pool_count": counts[STATUS_REJECT_POOL],
            "retain_ratio": counts[STATUS_RETAIN] / max(total_count, 1),
            "downweight_ratio": counts[STATUS_DOWNWEIGHT] / max(total_count, 1),
            "reject_candidate_ratio": counts[STATUS_REJECT_CANDIDATE] / max(total_count, 1),
            "reject_pool_ratio": counts[STATUS_REJECT_POOL] / max(total_count, 1),
        }
        for metric_name in metric_names:
            values = np.asarray(metric_values[metric_name], dtype=float)
            row[f"{metric_name}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric_name}_p25"] = float(np.percentile(values, 25)) if values.size else np.nan
            row[f"{metric_name}_p50"] = float(np.percentile(values, 50)) if values.size else np.nan
            row[f"{metric_name}_p75"] = float(np.percentile(values, 75)) if values.size else np.nan
        timeline_rows.append(row)

    timeline_csv = os.path.join(filter_dir, "filter_timeline_summary.csv")
    header = list(timeline_rows[0].keys())
    Path(checked_path(timeline_csv)).write_text(csv_text(header, timeline_rows), encoding="utf-8")

    epochs_array = np.asarray([row["epoch"] for row in timeline_rows], dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(epochs_array, [row["retain_count"] for row in timeline_rows], marker="o", label="retain")
    axes[0].plot(epochs_array, [row["downweight_count"] for row in timeline_rows], marker="o", label="downweight")
    axes[0].plot(
        epochs_array,
        [row["reject_candidate_count"] for row in timeline_rows],
        marker="o",
        label="reject_candidate",
    )
    axes[0].plot(epochs_array, [row["reject_pool_count"] for row in timeline_rows], marker="o", label="reject_pool")
    axes[0].set_ylabel("Sample Count")
    axes[0].set_title("Filtering status counts over epochs")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    score_specs = [
        ("reliability_score_mean", "reliability", "#111111"),
        ("waveform_quality_mean", "waveform_quality", "#1f77b4"),
        ("augmentation_consistency_mean", "augmentation_consistency", "#ff7f0e"),
        ("neighbor_agreement_mean", "neighbor_agreement", "#2ca02c"),
        ("epoch_stability_mean", "epoch_stability", "#d62728"),
        ("cluster_margin_mean", "cluster_margin", "#9467bd"),
    ]
    for key, label, color in score_specs:
        axes[1].plot(epochs_array, [row[key] for row in timeline_rows], marker="o", label=label, color=color)
    axes[1].axhline(low_threshold, color="#888888", linestyle="--", linewidth=1.0, label="low_threshold")
    axes[1].axhline(high_threshold, color="#222222", linestyle=":", linewidth=1.2, label="high_threshold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Filtering score means over epochs")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(filter_dir, "filter_timeline_overview.png"), dpi=200)
    plt.close(fig)


def save_filter_state(filter_dir: str, state: Dict, args):
    maybe_mkdir(filter_dir)
    epoch = int(state["epoch"])
    csv_path = os.path.join(filter_dir, f"filter_epoch_{epoch:03d}.csv")
    npz_path = os.path.join(filter_dir, f"filter_epoch_{epoch:03d}.npz")
    latest_npz_path = os.path.join(filter_dir, "latest_filter_state.npz")
    latest_json_path = os.path.join(filter_dir, "latest_filter_summary.json")

    header = [
        "sample_index",
        "status",
        "reliability_score",
        "waveform_quality_raw",
        "waveform_quality",
        "augmentation_consistency",
        "neighbor_agreement",
        "epoch_stability",
        "cluster_margin",
        "aligned_cluster_label",
        "low_score_count",
        "sample_weight",
        "in_reject_pool",
    ]
    rows = []
    for idx in range(len(state["reliability_score"])):
        rows.append(
            [
                idx,
                str(state["status"][idx]),
                float(state["reliability_score"][idx]),
                float(state["waveform_quality_raw"][idx]),
                float(state["waveform_quality"][idx]),
                float(state["augmentation_consistency"][idx]),
                float(state["neighbor_agreement"][idx]),
                float(state["epoch_stability"][idx]),
                float(state["cluster_margin"][idx]),
                int(state["aligned_cluster_labels"][idx]),
                int(state["low_score_count"][idx]),
                float(state["sample_weight"][idx]),
                int(bool(state["reject_pool_mask"][idx])),
            ]
        )
    Path(checked_path(csv_path)).write_text(csv_text(header, rows), encoding="utf-8")

    npz_payload = {
        "epoch": np.asarray(epoch, dtype=np.int64),
        "reliability_score": np.asarray(state["reliability_score"], dtype=np.float32),
        "waveform_quality_raw": np.asarray(state["waveform_quality_raw"], dtype=np.float32),
        "waveform_quality": np.asarray(state["waveform_quality"], dtype=np.float32),
        "augmentation_consistency": np.asarray(state["augmentation_consistency"], dtype=np.float32),
        "neighbor_agreement": np.asarray(state["neighbor_agreement"], dtype=np.float32),
        "epoch_stability": np.asarray(state["epoch_stability"], dtype=np.float32),
        "cluster_margin": np.asarray(state["cluster_margin"], dtype=np.float32),
        "aligned_cluster_labels": np.asarray(state["aligned_cluster_labels"], dtype=np.int64),
        "status_codes": np.asarray(state["status_codes"], dtype=np.int64),
        "low_score_count": np.asarray(state["low_score_count"], dtype=np.int64),
        "sample_weight": np.asarray(state["sample_weight"], dtype=np.float32),
        "reject_pool_mask": np.asarray(state["reject_pool_mask"], dtype=np.int64),
        "cluster_centers": np.asarray(state["cluster_centers"], dtype=np.float32),
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float32),
        "feature_scale": np.asarray(state["feature_scale"], dtype=np.float32),
        "filter_indices": np.asarray(state["filter_indices"], dtype=np.int64),
    }
    np.savez_compressed(checked_path(npz_path), **npz_payload)
    np.savez_compressed(checked_path(latest_npz_path), **npz_payload)

    counts = {
        STATUS_RETAIN: int(np.sum(state["status_codes"] == STATUS_TO_CODE[STATUS_RETAIN])),
        STATUS_DOWNWEIGHT: int(np.sum(state["status_codes"] == STATUS_TO_CODE[STATUS_DOWNWEIGHT])),
        STATUS_REJECT_CANDIDATE: int(
            np.sum(state["status_codes"] == STATUS_TO_CODE[STATUS_REJECT_CANDIDATE])
        ),
        STATUS_REJECT_POOL: int(np.sum(state["status_codes"] == STATUS_TO_CODE[STATUS_REJECT_POOL])),
    }
    summary = {
        "epoch": epoch,
        "counts": counts,
        "thresholds": {
            "low_score_threshold": float(args.low_score_threshold),
            "high_score_threshold": float(args.high_score_threshold),
            "low_score_patience": int(args.low_score_patience),
        },
        "score_summary": {
            "reliability_score": summarize_score_array(state["reliability_score"]),
            "waveform_quality": summarize_score_array(state["waveform_quality"]),
            "augmentation_consistency": summarize_score_array(state["augmentation_consistency"]),
            "neighbor_agreement": summarize_score_array(state["neighbor_agreement"]),
            "epoch_stability": summarize_score_array(state["epoch_stability"]),
            "cluster_margin": summarize_score_array(state["cluster_margin"]),
        },
    }
    save_json(latest_json_path, summary)
    export_filtering_timeline(
        filter_dir, float(args.low_score_threshold), float(args.high_score_threshold)
    )


def load_filter_state(stage1_dir: str) -> Optional[Dict]:
    npz_path = os.path.join(stage1_dir, "filtering", "latest_filter_state.npz")
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path, allow_pickle=False) as payload:
        state = {key: payload[key] for key in payload.files}
    state["epoch"] = int(np.asarray(state["epoch"]).item())
    state["status"] = np.asarray(
        [CODE_TO_STATUS[int(code)] for code in state["status_codes"]], dtype=object
    )
    state["reject_pool_mask"] = state["reject_pool_mask"].astype(bool)
    return state


def should_run_filter(epoch_index: int, args) -> bool:
    if not args.enable_periodic_filtering:
        return False
    if epoch_index <= args.warmup_epochs:
        return False
    return (epoch_index - args.warmup_epochs) % max(args.filter_interval, 1) == 0


def run_periodic_filtering(
    model, dataset: SeismicPolarityDataset, previous_state: Dict, args, epoch_index: int, stage1_dir: str
):
    filter_dir = os.path.join(stage1_dir, "filtering")
    active_mask = ~np.asarray(previous_state["reject_pool_mask"], dtype=bool)
    filter_indices = np.where(active_mask)[0]
    if filter_indices.size < 2:
        print(f"[filter] epoch={epoch_index}: skipped because active samples < 2.")
        return previous_state

    clean_waveforms = dataset.get_clean_waveforms(filter_indices)
    raw_features = encode_waveforms(
        model, clean_waveforms, batch_size=args.batch_size, cluster_feature=args.cluster_feature
    )
    features, feature_mean, feature_scale = preprocess_cluster_features(raw_features)
    raw_cluster_labels = cluster(features, n_clusters=2, method=args.stage1_cluster_method)
    cluster_centers = compute_binary_cluster_centers(features, raw_cluster_labels)

    aligned_labels, aligned_centers, instant_stability = align_binary_clusters_with_previous(
        previous_state.get("aligned_cluster_labels"),
        filter_indices,
        raw_cluster_labels,
        cluster_centers,
    )
    epoch_stability_full = np.asarray(previous_state["epoch_stability"], dtype=float).copy()
    prev_subset_stability = epoch_stability_full[filter_indices]
    epoch_stability_full[filter_indices] = (
        args.stability_ema_alpha * instant_stability
        + (1.0 - args.stability_ema_alpha) * prev_subset_stability
    )

    waveform_quality_raw, waveform_quality = compute_waveform_quality_scores(
        dataset,
        filter_indices,
    )
    augmentation_consistency = compute_filter_augmentation_consistency(
        model=model,
        clean_waveforms=clean_waveforms,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        cluster_centers=aligned_centers,
        views=args.n_tta_views,
        max_shift=args.tta_max_shift,
        batch_size_local=args.batch_size,
        cluster_feature=args.cluster_feature,
    )
    neighbor_agreement = compute_neighbor_agreement(
        features,
        aligned_labels,
        k=args.knn_k,
        metric="cosine",
    )
    cluster_margin_values = compute_cluster_margin(features, aligned_centers)
    reliability_score = fuse_reliability_scores(
        waveform_quality=waveform_quality,
        augmentation_consistency=augmentation_consistency,
        neighbor_agreement=neighbor_agreement,
        epoch_stability_values=epoch_stability_full[filter_indices],
        cluster_margin_values=cluster_margin_values,
        weight_dict={
            "w_q": args.w_q,
            "w_aug": args.w_aug,
            "w_knn": args.w_knn,
            "w_stab": args.w_stab,
            "w_margin": args.w_margin,
        },
    )

    status = np.asarray(previous_state["status"], dtype=object).copy()
    low_counts = np.asarray(previous_state["low_score_count"], dtype=np.int64).copy()
    reject_pool_mask = np.asarray(previous_state["reject_pool_mask"], dtype=bool).copy()
    reliability_full = np.asarray(previous_state["reliability_score"], dtype=float).copy()
    waveform_quality_raw_full = np.asarray(previous_state["waveform_quality_raw"], dtype=float).copy()
    waveform_quality_full = np.asarray(previous_state["waveform_quality"], dtype=float).copy()
    augmentation_consistency_full = np.asarray(
        previous_state["augmentation_consistency"], dtype=float
    ).copy()
    neighbor_agreement_full = np.asarray(previous_state["neighbor_agreement"], dtype=float).copy()
    cluster_margin_full = np.asarray(previous_state["cluster_margin"], dtype=float).copy()
    aligned_cluster_full = np.asarray(previous_state["aligned_cluster_labels"], dtype=np.int64).copy()
    sample_weight_full = np.asarray(previous_state["sample_weight"], dtype=float).copy()

    reliability_full[filter_indices] = reliability_score
    waveform_quality_raw_full[filter_indices] = waveform_quality_raw
    waveform_quality_full[filter_indices] = waveform_quality
    augmentation_consistency_full[filter_indices] = augmentation_consistency
    neighbor_agreement_full[filter_indices] = neighbor_agreement
    cluster_margin_full[filter_indices] = cluster_margin_values
    aligned_cluster_full[filter_indices] = aligned_labels

    status_subset, low_counts_subset, sample_weights_subset, reject_pool_subset = update_filter_membership(
        reliability_score=reliability_score,
        previous_low_counts=np.asarray(previous_state["low_score_count"], dtype=np.int64)[filter_indices],
        previous_reject_pool=np.asarray(previous_state["reject_pool_mask"], dtype=bool)[filter_indices],
        low_threshold=args.low_score_threshold,
        high_threshold=args.high_score_threshold,
        patience=args.low_score_patience,
        downweight_scale=args.downweight_loss_scale,
    )
    status[filter_indices] = np.asarray(status_subset, dtype=object)
    low_counts[filter_indices] = low_counts_subset
    sample_weight_full[filter_indices] = sample_weights_subset
    reject_pool_mask[filter_indices] = reject_pool_subset

    for idx in np.where(reject_pool_mask)[0]:
        status[idx] = STATUS_REJECT_POOL
        sample_weight_full[idx] = 0.0

    status_codes = np.asarray([STATUS_TO_CODE[str(item)] for item in status], dtype=np.int64)
    new_state = {
        "epoch": int(epoch_index),
        "reliability_score": reliability_full,
        "waveform_quality_raw": waveform_quality_raw_full,
        "waveform_quality": waveform_quality_full,
        "augmentation_consistency": augmentation_consistency_full,
        "neighbor_agreement": neighbor_agreement_full,
        "epoch_stability": epoch_stability_full,
        "cluster_margin": cluster_margin_full,
        "aligned_cluster_labels": aligned_cluster_full,
        "status": status,
        "status_codes": status_codes,
        "low_score_count": low_counts,
        "sample_weight": sample_weight_full,
        "reject_pool_mask": reject_pool_mask,
        "cluster_centers": aligned_centers,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "filter_indices": filter_indices,
    }

    save_filter_state(filter_dir, new_state, args)
    print(
        f"[filter] epoch={epoch_index} | retain={np.sum(status_codes == STATUS_TO_CODE[STATUS_RETAIN])} "
        f"| downweight={np.sum(status_codes == STATUS_TO_CODE[STATUS_DOWNWEIGHT])} "
        f"| reject_candidate={np.sum(status_codes == STATUS_TO_CODE[STATUS_REJECT_CANDIDATE])} "
        f"| reject_pool={np.sum(status_codes == STATUS_TO_CODE[STATUS_REJECT_POOL])}"
    )
    print_score_summary("reliability_score", reliability_score)
    print_score_summary("waveform_quality", waveform_quality)
    print_score_summary("augmentation_consistency", augmentation_consistency)
    print_score_summary("neighbor_agreement", neighbor_agreement)
    print_score_summary("epoch_stability", epoch_stability_full[filter_indices])
    print_score_summary("cluster_margin", cluster_margin_values)
    return new_state


# ==================== Encoding helpers ====================


def encode_from_model(model, batch_tensor, cluster_feature: str = "encoder"):
    if cluster_feature == "projector" and hasattr(model, "projector"):
        return model(batch_tensor)
    if hasattr(model, "encoder"):
        return model.encoder(batch_tensor)
    raise ValueError("Unsupported model for embedding extraction.")


def extract_raw_info(model, dataloader, eval_source: str = "clean", cluster_feature: str = "encoder"):
    model.eval()
    features = []
    original_waveforms = []
    polarities = []
    snrs = []
    with torch.no_grad():
        for _, (view1, view2, clean_wave, polarity, snr) in enumerate(dataloader):
            if eval_source == "clean":
                y = clean_wave.to(device)
            elif eval_source == "view1":
                y = view1.to(device)
            else:
                y = view2.to(device)
            feat = encode_from_model(model, y, cluster_feature=cluster_feature)
            features.append(feat.cpu().numpy())
            original_waveforms.append(clean_wave.cpu().numpy())
            polarities.append(polarity.cpu().numpy())
            snrs.append(snr.cpu().numpy())
    return (
        np.vstack(features),
        np.vstack(original_waveforms),
        np.concatenate(polarities),
        np.concatenate(snrs),
    )


def extract_info(model, dataloader, eval_source: str = "clean", cluster_feature: str = "encoder"):
    raw_features, original_waveforms, polarities, snrs = extract_raw_info(
        model, dataloader, eval_source=eval_source, cluster_feature=cluster_feature
    )
    features, _, _ = preprocess_cluster_features(raw_features)
    return features, original_waveforms, polarities, snrs


def encode_waveforms(model, waveforms, batch_size=256, cluster_feature: str = "encoder"):
    waves = np.asarray(waveforms, dtype=np.float32)
    if waves.ndim == 2:
        waves = waves[:, None, :]
    elif waves.ndim != 3 or waves.shape[1] != 1:
        raise ValueError(
            "waveforms must have shape (n_samples, n_points) or (n_samples, 1, n_points)."
        )
    features = []
    model.eval()
    with torch.no_grad():
        for start in range(0, waves.shape[0], batch_size):
            batch = torch.from_numpy(waves[start : start + batch_size]).to(device)
            feat = encode_from_model(model, batch, cluster_feature=cluster_feature)
            features.append(feat.cpu().numpy())
    return np.vstack(features)


# ==================== Visualization ====================


def visualize_clusters_simple(emb, preds, score=None, out_dir="results", ver="pca", filename=None):
    maybe_mkdir(out_dir)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=preds, cmap="viridis", s=5, alpha=0.4)
    plt.colorbar(scatter, ax=ax)
    ax.set_title("Predicted Clusters")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if score is None:
        fig.suptitle(f"[{ver.upper()}]")
    else:
        fig.suptitle(f"[{ver.upper()}] | Score: {score:.4f}")
    filename = filename or f"clustering_visualization_{ver}.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close(fig)


def visualize_clusters_compare(
    emb,
    preds,
    true_labels,
    acc_value=np.nan,
    acc_value_excluding_label2=np.nan,
    score=None,
    out_dir="results",
    ver="pca",
    filename=None,
):
    maybe_mkdir(out_dir)
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    scatter_pred = ax[0].scatter(emb[:, 0], emb[:, 1], c=preds, cmap="viridis", s=5, alpha=0.4)
    scatter_true = ax[1].scatter(emb[:, 0], emb[:, 1], c=true_labels, cmap="viridis", s=5, alpha=0.4)
    plt.colorbar(scatter_pred, ax=ax[0])
    plt.colorbar(scatter_true, ax=ax[1])
    ax[0].set_title("Predicted Clusters")
    ax[1].set_title("True Labels")
    for axis in ax:
        axis.set_xlabel("Component 1")
        axis.set_ylabel("Component 2")
    title = f"[{ver.upper()}] | ACC={acc_value:.4f}"
    if np.isfinite(acc_value_excluding_label2):
        title += f" | ACC(no true=2)={acc_value_excluding_label2:.4f}"
    if score is not None and np.isfinite(score):
        title += f" | silhouette={score:.4f}"
    fig.suptitle(title)
    filename = filename or f"clustering_visualization_{ver}_true_compare.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close(fig)


def visualize_training_item(
    dataset, sample_index: int, pred_label: int, save_dir: str, true_label=None, extra_text: str = ""
):
    maybe_mkdir(save_dir)
    clean_wave = dataset.get_clean_waveform(int(sample_index))
    full_wave = np.asarray(dataset.waveforms[int(sample_index)], dtype=float)
    snr = float(dataset.snrs[int(sample_index)])
    if true_label is None:
        true_label = int(dataset.polarities[int(sample_index)])

    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    ax[0].plot(clean_wave)
    title = f"true: {true_label}, pred: {pred_label}, snr: {snr:.2f}"
    if extra_text:
        title += f", {extra_text}"
    ax[0].set_title(title)
    ax[1].plot(full_wave)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"sample_{int(sample_index)}_cluster_{int(pred_label)}.png"))
    plt.close(fig)


def plot_training_losses(train_losses, on_diag_losses, off_diag_losses, out_path):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Total Loss")
    plt.plot(on_diag_losses, label="On-Diagonal Loss")
    plt.plot(off_diag_losses, label="Off-Diagonal Loss")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Barlow Twins Training Loss")
    plt.grid(True)
    plt.savefig(out_path, dpi=300)
    plt.close()


# ==================== Stage 1 training ====================


def train_barlow_twins(model, dataset, optimizer, scheduler, args, save_dir):
    train_losses = []
    on_diag_losses = []
    off_diag_losses = []
    acc_dir = {cm: [] for cm in ["kmeans", "dbscan", "agglomerative", "spectral", "km_cosine", "agg_cosine"]}
    eval_loader = build_eval_dataloader(dataset, batch_size=args.batch_size)
    filter_state = init_filter_state(len(dataset))

    maybe_mkdir(os.path.join(save_dir, "train_out"))
    maybe_mkdir(os.path.join(save_dir, "filtering"))

    for epoch_idx in range(args.epochs):
        epoch_one_based = epoch_idx + 1
        if should_run_filter(epoch_one_based, args):
            filter_state = run_periodic_filtering(
                model=model,
                dataset=dataset,
                previous_state=filter_state,
                args=args,
                epoch_index=epoch_one_based,
                stage1_dir=save_dir,
            )

        sample_weights = filter_state["sample_weight"] if args.enable_periodic_filtering else None
        train_loader = build_train_dataloader(
            dataset,
            sample_weights=sample_weights,
            batch_size=args.batch_size,
            drop_last=True,
        )

        model.train()
        epoch_loss = 0.0
        epoch_on_diag_loss = 0.0
        epoch_off_diag_loss = 0.0
        n_total = 0
        progress = tqdm.tqdm(train_loader, desc=f"E {epoch_one_based}/{args.epochs}", unit="batch")
        for view1, view2, *_ in progress:
            view1, view2 = view1.to(device), view2.to(device)
            z1 = model(view1)
            z2 = model(view2)
            on_diag_loss, off_diag_loss = model.compute_loss(z1, z2)
            loss = on_diag_loss + args.lambda_param * off_diag_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_on_diag_loss += on_diag_loss.item()
            epoch_off_diag_loss += off_diag_loss.item()
            n_total += view1.size(0)
            progress.set_postfix({"loss": loss.item() / view1.size(0), "lr": scheduler.get_last_lr()[0]})

        scheduler.step()
        train_losses.append(epoch_loss / max(n_total, 1))
        on_diag_losses.append(epoch_on_diag_loss / max(n_total, 1))
        off_diag_losses.append(epoch_off_diag_loss / max(n_total, 1) * args.lambda_param)

        if args.eval_interval and epoch_one_based % int(args.eval_interval) == 0:
            features, _, polarities, _ = extract_info(
                model,
                eval_loader,
                eval_source=args.eval_source,
                cluster_feature=args.cluster_feature,
            )
            for method in acc_dir:
                try:
                    cluster_labels = cluster(features, n_clusters=2, method=method)
                    acc_dir[method].append(acc(polarities, cluster_labels)[0])
                except Exception:
                    acc_dir[method].append(np.nan)
            eval_epochs = [
                (i + 1) * int(args.eval_interval or 5) for i in range(len(acc_dir[list(acc_dir)[0]]))
            ]
            for idx in range(2):
                plt.figure(figsize=(10, 6))
                for method in acc_dir:
                    plt.plot(eval_epochs, acc_dir[method], label=method)
                if idx == 0:
                    plt.ylim(0.85, 0.99)
                plt.xlabel("Epoch")
                plt.ylabel("Clustering Accuracy")
                plt.title("Clustering Accuracy over Epochs")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{save_dir}/train_out/clustering_accuracy_{idx}.png", dpi=300)
                plt.close()

        plot_training_losses(
            train_losses,
            on_diag_losses,
            off_diag_losses,
            out_path=os.path.join(save_dir, "train_out", "training_loss.png"),
        )

    torch.save({"model_state_dict": model.state_dict()}, os.path.join(save_dir, "final_pwave_model.pth"))
    if args.enable_periodic_filtering:
        save_filter_state(os.path.join(save_dir, "filtering"), filter_state, args)
    return {
        "train_losses": train_losses,
        "on_diag_losses": on_diag_losses,
        "off_diag_losses": off_diag_losses,
        "acc_dir": acc_dir,
        "filter_state": filter_state,
    }


def summarize_stage1(model, dataset, args, stage1_dir, filter_state):
    eval_loader = build_eval_dataloader(dataset, batch_size=args.batch_size)
    features, _, polarities, _ = extract_info(
        model,
        eval_loader,
        eval_source=args.eval_source,
        cluster_feature=args.cluster_feature,
    )
    summary = {
        "stage": "stage1",
        "save_path": stage1_dir,
        "cluster_method": args.stage1_cluster_method,
        "filtering_enabled": bool(args.enable_periodic_filtering),
    }
    for method in ["kmeans", "spectral"]:
        labels = cluster(features, n_clusters=2, method=method)
        summary[f"{method}_acc"] = float(acc(polarities, labels)[0])
        summary[f"{method}_silhouette"] = float(sklearn.metrics.silhouette_score(features, labels))
    if filter_state is not None:
        for name, code in STATUS_TO_CODE.items():
            summary[f"count_{name}"] = int(np.sum(filter_state["status_codes"] == code))
        summary["mean_reliability"] = float(np.mean(filter_state["reliability_score"]))
    save_json(os.path.join(stage1_dir, "stage1_summary.json"), summary)

    csv_path = os.path.join(stage1_dir, "training_case_comparison.csv")
    text = csv_text(list(summary.keys()), [list(summary.values())])
    Path(checked_path(csv_path)).write_text(text, encoding="utf-8")


# ==================== Cluster result export ====================


def export_cluster_results(
    dataset: SeismicPolarityDataset,
    sample_indices: np.ndarray,
    features: np.ndarray,
    cluster_labels: np.ndarray,
    save_dir: str,
    prefix: str,
    method: str,
    status_lookup: Optional[np.ndarray] = None,
):
    maybe_mkdir(save_dir)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    true_labels = np.asarray(dataset.polarities[sample_indices], dtype=np.int64)
    snrs = np.asarray(dataset.snrs[sample_indices], dtype=float)
    has_true_labels = bool(np.any(np.isin(true_labels, (0, 1, 2))))
    if has_true_labels:
        acc_value, _, _, mapped_preds = acc(true_labels, cluster_labels)
        binary_mask = true_labels != 2
        if np.any(binary_mask):
            acc_value_excluding_label2 = acc(true_labels[binary_mask], cluster_labels[binary_mask])[0]
        else:
            acc_value_excluding_label2 = np.nan
    else:
        acc_value = np.nan
        acc_value_excluding_label2 = np.nan
        binary_mask = np.zeros(len(true_labels), dtype=bool)
        mapped_preds = np.full(len(cluster_labels), -1, dtype=np.int64)

    score = np.nan
    if len(np.unique(cluster_labels)) > 1 and len(cluster_labels) > 1:
        try:
            score = float(sklearn.metrics.silhouette_score(features, cluster_labels))
        except Exception:
            score = np.nan

    csv_path = os.path.join(save_dir, f"cluster_results_{method}.csv")
    header = [
        "sample_index",
        "snr",
        "true_polarity",
        "pred_cluster",
        "pred_cluster_mapped_to_true_for_analysis",
        "is_correct_after_mapping",
    ]
    if status_lookup is not None:
        header.append("status")
    rows = []
    for local_idx, sample_index in enumerate(sample_indices):
        row = [
            int(sample_index),
            float(snrs[local_idx]),
            int(true_labels[local_idx]),
            int(cluster_labels[local_idx]),
            int(mapped_preds[local_idx]),
            int(mapped_preds[local_idx] == true_labels[local_idx]) if has_true_labels else -1,
        ]
        if status_lookup is not None:
            row.append(str(status_lookup[int(sample_index)]))
        rows.append(row)
    Path(checked_path(csv_path)).write_text(csv_text(header, rows), encoding="utf-8")

    emb_tsne = dedim(features, ver="tsne")
    emb_pca = dedim(features, ver="pca")
    visualize_clusters_simple(
        emb_tsne,
        preds=cluster_labels,
        score=score,
        out_dir=save_dir,
        ver="tsne",
        filename="clustering_visualization_tsne.png",
    )
    visualize_clusters_simple(
        emb_pca,
        preds=cluster_labels,
        score=score,
        out_dir=save_dir,
        ver="pca",
        filename="clustering_visualization_pca.png",
    )
    if has_true_labels:
        visualize_clusters_compare(
            emb_tsne,
            preds=mapped_preds,
            true_labels=true_labels,
            acc_value=acc_value,
            acc_value_excluding_label2=acc_value_excluding_label2,
            score=score,
            out_dir=save_dir,
            ver="tsne",
            filename="clustering_visualization_tsne_true_compare.png",
        )
        visualize_clusters_compare(
            emb_pca,
            preds=mapped_preds,
            true_labels=true_labels,
            acc_value=acc_value,
            acc_value_excluding_label2=acc_value_excluding_label2,
            score=score,
            out_dir=save_dir,
            ver="pca",
            filename="clustering_visualization_pca_true_compare.png",
        )
        save_confusion_matrix_percent(
            true_labels=true_labels,
            pred_labels=np.where(mapped_preds >= 0, mapped_preds, 2),
            class_labels=[0, 1, 2],
            out_dir=save_dir,
            filename_prefix="confusion_matrix_percent_all012",
            title="Confusion Matrix (%) | Labels 0/1/2",
        )
        if np.any(binary_mask):
            save_confusion_matrix_percent(
                true_labels=true_labels[binary_mask],
                pred_labels=mapped_preds[binary_mask],
                class_labels=[0, 1],
                out_dir=save_dir,
                filename_prefix="confusion_matrix_percent_binary01",
                title="Confusion Matrix (%) | Labels 0/1 only",
            )

    rng = np.random.default_rng(42)
    for cluster_id in np.unique(cluster_labels):
        cluster_member_indices = sample_indices[cluster_labels == cluster_id]
        if cluster_member_indices.size == 0:
            continue
        n_show = min(50, cluster_member_indices.size)
        selected = rng.choice(cluster_member_indices, size=n_show, replace=False)
        cluster_dir = os.path.join(save_dir, f"visual_cluster_{method}_{int(cluster_id)}")
        maybe_mkdir(cluster_dir)
        for old_file in glob.glob(os.path.join(cluster_dir, "*")):
            if os.path.isfile(old_file):
                os.unlink(old_file)
        for sample_index in selected:
            extra_text = ""
            if status_lookup is not None:
                extra_text = f"status: {status_lookup[int(sample_index)]}"
            visualize_training_item(
                dataset,
                sample_index=int(sample_index),
                pred_label=int(cluster_id),
                save_dir=cluster_dir,
                true_label=int(dataset.polarities[int(sample_index)]),
                extra_text=extra_text,
            )
    print(f"Saved {prefix} cluster csv to: {csv_path}")
    print(
        f"{prefix} clustering_acc_all012_analysis="
        f"{acc_value:.4f}"
    )
    if np.isfinite(acc_value_excluding_label2):
        print(
            f"{prefix} clustering_acc_binary01_analysis="
            f"{acc_value_excluding_label2:.4f}"
        )


def export_stage1_cluster_artifacts(model, dataset, args, stage1_dir, filter_state):
    eval_loader = build_eval_dataloader(dataset, batch_size=args.batch_size)
    raw_features, _, _, _ = extract_raw_info(
        model,
        eval_loader,
        eval_source=args.eval_source,
        cluster_feature=args.cluster_feature,
    )
    features, _, _ = preprocess_cluster_features(raw_features)
    cluster_labels = cluster(features, n_clusters=2, method=args.stage1_cluster_method)

    if filter_state is not None and np.any(np.asarray(filter_state["aligned_cluster_labels"]) >= 0):
        aligned_prev = np.asarray(filter_state["aligned_cluster_labels"], dtype=np.int64)
        full_indices = np.arange(len(dataset), dtype=np.int64)
        cluster_centers = compute_binary_cluster_centers(features, cluster_labels)
        aligned_labels, _, _ = align_binary_clusters_with_previous(
            aligned_prev,
            full_indices,
            cluster_labels,
            cluster_centers,
        )
        cluster_labels = aligned_labels

    export_cluster_results(
        dataset=dataset,
        sample_indices=np.arange(len(dataset), dtype=np.int64),
        features=features,
        cluster_labels=cluster_labels,
        save_dir=os.path.join(stage1_dir, "cluster_results"),
        prefix="stage1",
        method=args.stage1_cluster_method,
        status_lookup=None if filter_state is None else np.asarray(filter_state["status"], dtype=object),
    )


def export_stage2_cluster_artifacts(
    dataset: SeismicPolarityDataset,
    selected_indices: np.ndarray,
    features: np.ndarray,
    pseudo_labels: np.ndarray,
    stage2_dir: str,
    method: str,
    status_lookup: Optional[np.ndarray] = None,
):
    export_cluster_results(
        dataset=dataset,
        sample_indices=selected_indices,
        features=features,
        cluster_labels=pseudo_labels,
        save_dir=os.path.join(stage2_dir, "cluster_results"),
        prefix="stage2",
        method=method,
        status_lookup=status_lookup,
    )


# ==================== Stage 2 training ====================


def train_stage2_pseudo_classifier(stage1_dir: str, args):
    config_path = os.path.join(stage1_dir, "stage1_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing Stage 1 config: {config_path}")

    stage1_config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    dataset = dataset_from_stage1_config(stage1_config)

    checkpoint_path = os.path.join(stage1_dir, "final_pwave_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing Stage 1 checkpoint: {checkpoint_path}")

    stage1_model = build_barlow_model(
        base_channels=int(stage1_config["base_cha"]),
        projector_dims=stage1_config["projector_dims"],
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    stage1_model.load_state_dict(checkpoint["model_state_dict"])
    stage1_model.to(device)
    stage1_model.eval()

    if not args.batch_size:
        args.batch_size = int(stage1_config["batch_size"])

    filter_state = load_filter_state(stage1_dir)
    if filter_state is None:
        status_codes = np.full(len(dataset), STATUS_TO_CODE[STATUS_RETAIN], dtype=np.int64)
        source_weights = np.ones(len(dataset), dtype=float)
        previous_cluster_labels = None
    else:
        status_codes = np.asarray(filter_state["status_codes"], dtype=np.int64)
        source_weights = np.asarray(filter_state["sample_weight"], dtype=float)
        previous_cluster_labels = np.asarray(filter_state["aligned_cluster_labels"], dtype=np.int64)

    if args.stage2_include_downweight:
        selected_mask = np.isin(
            status_codes,
            [STATUS_TO_CODE[STATUS_RETAIN], STATUS_TO_CODE[STATUS_DOWNWEIGHT]],
        )
    else:
        selected_mask = status_codes == STATUS_TO_CODE[STATUS_RETAIN]
    selected_indices = np.where(selected_mask)[0]
    if selected_indices.size < 2:
        raise ValueError("Not enough samples left for Stage 2 pseudo-label training.")

    clean_waveforms = dataset.get_clean_waveforms(selected_indices)
    raw_features = encode_waveforms(
        stage1_model,
        clean_waveforms,
        batch_size=args.batch_size,
        cluster_feature=str(stage1_config.get("cluster_feature", "encoder")),
    )
    features, feature_mean, feature_scale = preprocess_cluster_features(raw_features)
    pseudo_labels = cluster(features, n_clusters=2, method=args.stage2_cluster_method)
    cluster_centers = compute_binary_cluster_centers(features, pseudo_labels)

    aligned_labels, aligned_centers, _ = align_binary_clusters_with_previous(
        previous_cluster_labels,
        selected_indices,
        pseudo_labels,
        cluster_centers,
    )
    pseudo_labels = aligned_labels
    cluster_centers = aligned_centers

    stage2_dir = args.stage2_save_dir or os.path.join(os.path.dirname(stage1_dir), "stage2")
    maybe_mkdir(stage2_dir)
    maybe_mkdir(os.path.join(stage2_dir, "train_out"))

    pseudo_csv_path = os.path.join(stage2_dir, "pseudo_labels.csv")
    pseudo_rows = []
    for local_idx, sample_index in enumerate(selected_indices):
        status_name = CODE_TO_STATUS[int(status_codes[sample_index])]
        pseudo_rows.append(
            [
                int(sample_index),
                int(pseudo_labels[local_idx]),
                f"cluster_{chr(ord('A') + int(pseudo_labels[local_idx]))}",
                status_name,
                float(source_weights[sample_index]),
            ]
        )
    Path(checked_path(pseudo_csv_path)).write_text(
        csv_text(
            ["sample_index", "pseudo_label", "pseudo_label_name", "source_status", "sample_weight"],
            pseudo_rows,
        ),
        encoding="utf-8",
    )

    pseudo_dataset = PseudoLabelDataset(
        base_dataset=dataset,
        indices=selected_indices,
        pseudo_labels=pseudo_labels,
        sample_weights=np.where(
            source_weights[selected_indices] > 0, source_weights[selected_indices], 1.0
        ),
    )
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    classifier = build_stage2_classifier(
        base_channels=int(stage1_config["base_cha"]),
        head_hidden_dim=args.stage2_head_hidden_dim,
    )
    classifier.encoder.load_state_dict(stage1_model.encoder.state_dict())
    classifier.to(device)

    if args.whether_freeze_encoder_in_stage2:
        for parameter in classifier.encoder.parameters():
            parameter.requires_grad = False

    optimizer = optim.AdamW(
        [param for param in classifier.parameters() if param.requires_grad],
        lr=args.stage2_lr,
        weight_decay=args.stage2_weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.stage2_epochs, 1))

    losses = []
    accuracies = []
    for epoch_idx in range(args.stage2_epochs):
        classifier.train()
        if args.whether_freeze_encoder_in_stage2:
            classifier.encoder.eval()
        epoch_losses = []
        correct = 0
        total = 0
        progress = tqdm.tqdm(pseudo_loader, desc=f"Stage2 {epoch_idx + 1}/{args.stage2_epochs}", unit="batch")
        for clean_wave, labels, sample_weight_tensor, _ in progress:
            clean_wave = clean_wave.to(device)
            labels = labels.to(device)
            sample_weight_tensor = sample_weight_tensor.to(device)

            logits = classifier(clean_wave)
            loss_per_sample = F.cross_entropy(logits, labels, reduction="none")
            loss = (loss_per_sample * sample_weight_tensor).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            progress.set_postfix({"loss": loss.item(), "acc": correct / max(total, 1)})

        scheduler.step()
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else np.nan)
        accuracies.append(correct / max(total, 1))

    plt.figure(figsize=(10, 4))
    plt.plot(losses, label="Stage2 Loss")
    plt.plot(accuracies, label="Stage2 Pseudo Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.title("Stage 2 Pseudo-Label Training")
    plt.savefig(os.path.join(stage2_dir, "train_out", "stage2_training.png"), dpi=300)
    plt.close()

    stage2_checkpoint_path = os.path.join(stage2_dir, "stage2_cluster_classifier.pth")
    torch.save(
        {
            "encoder_state_dict": classifier.encoder.state_dict(),
            "head_state_dict": classifier.head.state_dict(),
            "head_hidden_dim": int(args.stage2_head_hidden_dim),
            "cluster_centers": np.asarray(cluster_centers, dtype=np.float32),
            "feature_mean": np.asarray(feature_mean, dtype=np.float32),
            "feature_scale": np.asarray(feature_scale, dtype=np.float32),
            "selected_indices": np.asarray(selected_indices, dtype=np.int64),
            "pseudo_labels": np.asarray(pseudo_labels, dtype=np.int64),
            "stage1_config": stage1_config,
            "stage2_config": dict(vars(args)),
            "note": "This is a cluster A/B pseudo-label model, not an up/down semantic model.",
        },
        stage2_checkpoint_path,
    )

    save_json(
        os.path.join(stage2_dir, "stage2_summary.json"),
        {
            "stage": "stage2",
            "stage1_dir": stage1_dir,
            "stage2_dir": stage2_dir,
            "pseudo_label_file": pseudo_csv_path,
            "checkpoint_path": stage2_checkpoint_path,
            "selected_sample_count": int(len(selected_indices)),
            "freeze_encoder": bool(args.whether_freeze_encoder_in_stage2),
            "include_downweight": bool(args.stage2_include_downweight),
            "cluster_method": args.stage2_cluster_method,
            "final_loss": float(losses[-1]) if losses else np.nan,
            "final_pseudo_accuracy": float(accuracies[-1]) if accuracies else np.nan,
            "note": "Pseudo labels are cluster A/B only.",
        },
    )
    export_stage2_cluster_artifacts(
        dataset=dataset,
        selected_indices=selected_indices,
        features=features,
        pseudo_labels=pseudo_labels,
        stage2_dir=stage2_dir,
        method=args.stage2_cluster_method,
        status_lookup=np.asarray(
            [CODE_TO_STATUS[int(code)] for code in status_codes],
            dtype=object,
        ),
    )
    print(f"Saved Stage 2 checkpoint to: {stage2_checkpoint_path}")
    print("Stage 2 output is cluster A/B only, not up/down semantics.")


# ==================== Eval (artifact refresh) ====================


def eval_stage1_artifacts(stage1_dir: str):
    config_path = os.path.join(stage1_dir, "stage1_config.json")
    checkpoint_path = os.path.join(stage1_dir, "final_pwave_model.pth")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing Stage 1 config: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing Stage 1 checkpoint: {checkpoint_path}")

    stage1_config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    dataset = dataset_from_stage1_config(stage1_config)
    model = build_barlow_model(
        base_channels=int(stage1_config["base_cha"]),
        projector_dims=stage1_config["projector_dims"],
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    filter_state = load_filter_state(stage1_dir)
    export_stage1_cluster_artifacts(
        model=model,
        dataset=dataset,
        args=SimpleNamespace(
            batch_size=int(stage1_config["batch_size"]),
            stage1_cluster_method=str(stage1_config["stage1_cluster_method"]),
            eval_source=str(stage1_config.get("eval_source", "clean")),
            cluster_feature=str(stage1_config.get("cluster_feature", "encoder")),
        ),
        stage1_dir=stage1_dir,
        filter_state=filter_state,
    )
    export_filtering_timeline(
        os.path.join(stage1_dir, "filtering"),
        float(stage1_config["low_score_threshold"]),
        float(stage1_config["high_score_threshold"]),
    )
    print(f"Stage 1 eval artifacts refreshed under: {os.path.join(stage1_dir, 'cluster_results')}")


def eval_stage2_artifacts(stage2_dir: str):
    checkpoint_path = os.path.join(stage2_dir, "stage2_cluster_classifier.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing Stage 2 checkpoint: {checkpoint_path}")

    bundle = load_checkpoint(checkpoint_path, map_location=device)
    stage1_config = bundle["stage1_config"]

    dataset = dataset_from_stage1_config(stage1_config)

    model = build_stage2_classifier(
        base_channels=int(stage1_config["base_cha"]),
        head_hidden_dim=int(bundle.get("head_hidden_dim", 0)),
    )
    model.encoder.load_state_dict(bundle["encoder_state_dict"])
    model.head.load_state_dict(bundle["head_state_dict"])
    model.to(device)
    model.eval()

    selected_indices = np.asarray(bundle["selected_indices"], dtype=np.int64)
    pseudo_labels = np.asarray(bundle["pseudo_labels"], dtype=np.int64)
    feature_mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(bundle["feature_scale"], dtype=np.float32)

    clean_waveforms = dataset.get_clean_waveforms(selected_indices)
    raw_features = encode_waveforms(
        model,
        clean_waveforms,
        batch_size=int(stage1_config["batch_size"]),
        cluster_feature=str(stage1_config.get("cluster_feature", "encoder")),
    )
    features, _, _ = preprocess_cluster_features(
        raw_features,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )

    stage1_dir = os.path.dirname(stage2_dir.rstrip("/"))
    stage1_filter_state = load_filter_state(os.path.join(stage1_dir, "stage1"))
    if stage1_filter_state is None:
        status_lookup = None
    else:
        status_lookup = np.asarray(stage1_filter_state["status"], dtype=object)

    stage2_config = bundle.get("stage2_config", {})
    method = str(stage2_config.get("stage2_cluster_method", "spectral"))
    export_stage2_cluster_artifacts(
        dataset=dataset,
        selected_indices=selected_indices,
        features=features,
        pseudo_labels=pseudo_labels,
        stage2_dir=stage2_dir,
        method=method,
        status_lookup=status_lookup,
    )
    print(f"Stage 2 eval artifacts refreshed under: {os.path.join(stage2_dir, 'cluster_results')}")


# ==================== CLI ====================


def main_stage1(args):
    cfg = resolve_stage1_config(args)
    if not cfg.get("data_path"):
        raise ValueError("No training data given. Pass --data-path or set data_path in --config.")
    data_path = resolve_data_path(cfg["data_path"])
    args_ns = SimpleNamespace(**cfg)

    stage1_dir = args.save_path or build_default_save_path(cfg)
    maybe_mkdir(stage1_dir)
    maybe_mkdir(os.path.join(stage1_dir, "train_out"))
    maybe_mkdir(os.path.join(stage1_dir, "filtering"))

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    loader_kwargs = dict(
        data_path=data_path,
        snr_range=cfg["snr_range"],
        n_select=cfg["num_used"],
        resize=cfg["resize"],
        shift=cfg["shift"],
        aug_shift=cfg["aug_shift"],
        aug_noise_std_range=cfg["aug_noise_std_range"],
        aug_scale_range=cfg["aug_scale_range"],
        norm_mod=cfg["norm_mod"],
    )
    if cfg["dataset_source"] == "ridgecrest_unlabeled":
        dataset = load_ridgecrest_unlabeled_dataset(**loader_kwargs)
    else:
        dataset = load_scsn_polarity_dataset(
            bino=cfg["bino"],
            waveform_key=cfg["waveform_key"],
            label_key=cfg["label_key"],
            snr_key=cfg["snr_key"],
            selection_seed=cfg["data_selection_seed"],
            **loader_kwargs,
        )

    model = build_barlow_model(
        base_channels=cfg["base_cha"], projector_dims=cfg["projector_dims"]
    )
    sample_loader = build_train_dataloader(dataset, batch_size=cfg["batch_size"], drop_last=True)
    sampleA, *_ = next(iter(sample_loader))
    with torch.no_grad():
        z1 = model(sampleA.to(device))
        print(f"Sample input shape: {sampleA.shape}")
        print(f"Sample feature shape: {z1.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg["epochs"], 1))
    training_result = train_barlow_twins(model, dataset, optimizer, scheduler, args_ns, stage1_dir)

    save_json(os.path.join(stage1_dir, "stage1_config.json"), stage1_runtime_config(cfg, data_path))
    summarize_stage1(model, dataset, args_ns, stage1_dir, training_result["filter_state"])
    export_stage1_cluster_artifacts(
        model=model,
        dataset=dataset,
        args=args_ns,
        stage1_dir=stage1_dir,
        filter_state=training_result["filter_state"],
    )
    print(f"Stage 1 finished. Output dir: {stage1_dir}")


def main_stage2(args):
    cfg = resolve_stage2_config(args)
    train_stage2_pseudo_classifier(stage1_dir=cfg["stage1_dir"], args=SimpleNamespace(**cfg))


def main_eval(args):
    stage2_dir = args.stage2_dir or os.path.join(
        os.path.dirname(str(args.stage1_dir).rstrip("/")), "stage2"
    )
    if args.target in {"stage1", "both"}:
        eval_stage1_artifacts(args.stage1_dir)
    if args.target in {"stage2", "both"}:
        eval_stage2_artifacts(stage2_dir)


def build_stage1_parser(subparsers):
    parser = subparsers.add_parser("stage1", help="Run Stage 1 anchor-free representation learning.")
    parser.add_argument("--config", default=None, help="YAML config file; CLI flags override it.")
    parser.add_argument("--save-path", default=None, help="Output directory (default: auto-named).")
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--dataset-source",
        default=None,
        choices=["scsn", "ridgecrest_unlabeled"],
        help=(
            "scsn: labelled SCSN-style HDF5 (X/Y/snr keys); ridgecrest_unlabeled: "
            "unlabelled Ridgecrest phasenet group, trained without labels (implies "
            "no bino filtering)."
        ),
    )
    parser.add_argument("--waveform-key", default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--snr-key", default=None)
    parser.add_argument("--snr-min", type=float, default=None)
    parser.add_argument("--snr-max", type=float, default=None)
    parser.add_argument("--num-used", type=int, default=None)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--shift", type=int, default=None)
    parser.add_argument("--bino", action="store_true", default=None)
    parser.add_argument("--no-bino", action="store_false", dest="bino", help="Keep label-2 samples.")
    parser.add_argument("--base-channels", type=int, default=None, dest="base_cha")
    parser.add_argument("--projector-dims", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-param", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--cluster-feature", default=None, choices=["encoder", "projector"])
    parser.add_argument("--norm-mod", default=None, choices=["rms", "max", "bn"])
    parser.add_argument("--eval-source", default=None, choices=["clean", "view1"])
    parser.add_argument("--aug-shift", type=int, default=None)
    parser.add_argument("--aug-noise-min", type=float, default=None)
    parser.add_argument("--aug-noise-max", type=float, default=None)
    parser.add_argument("--aug-scale-min", type=float, default=None)
    parser.add_argument("--aug-scale-max", type=float, default=None)
    parser.add_argument("--stage1-cluster-method", default=None)
    parser.add_argument("--enable-periodic-filtering", action="store_true", default=None)
    parser.add_argument(
        "--disable-periodic-filtering", action="store_false", dest="enable_periodic_filtering"
    )
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--filter-interval", type=int, default=None)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Run the diagnostic clustering-accuracy eval every N epochs; 0 disables it.",
    )
    parser.add_argument("--n-tta-views", type=int, default=None)
    parser.add_argument("--tta-max-shift", type=int, default=None)
    parser.add_argument("--knn-k", type=int, default=None)
    parser.add_argument("--low-score-threshold", type=float, default=None)
    parser.add_argument("--high-score-threshold", type=float, default=None)
    parser.add_argument("--low-score-patience", type=int, default=None)
    parser.add_argument("--w-q", type=float, default=None)
    parser.add_argument("--w-aug", type=float, default=None)
    parser.add_argument("--w-knn", type=float, default=None)
    parser.add_argument("--w-stab", type=float, default=None)
    parser.add_argument("--w-margin", type=float, default=None)
    parser.add_argument("--downweight-loss-scale", type=float, default=None)
    parser.add_argument("--stability-ema-alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.set_defaults(func=main_stage1)


def build_stage2_parser(subparsers):
    parser = subparsers.add_parser("stage2", help="Run Stage 2 anchor-free pseudo-label training.")
    parser.add_argument("--stage1-dir", required=True, help="Directory holding stage1_config.json.")
    parser.add_argument("--stage2-save-dir", default=None)
    parser.add_argument("--stage2-cluster-method", default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--stage2-lr", type=float, default=None)
    parser.add_argument("--stage2-weight-decay", type=float, default=None)
    parser.add_argument("--stage2-include-downweight", action="store_true", default=None)
    parser.add_argument(
        "--stage2-only-retain", action="store_false", dest="stage2_include_downweight"
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=None,
        dest="whether_freeze_encoder_in_stage2",
    )
    parser.add_argument(
        "--finetune-encoder",
        action="store_false",
        default=None,
        dest="whether_freeze_encoder_in_stage2",
    )
    parser.add_argument("--stage2-head-hidden-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.set_defaults(func=main_stage2)


def build_eval_parser(subparsers):
    parser = subparsers.add_parser(
        "eval", help="Rebuild Stage 1/2 cluster visualizations without retraining."
    )
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--stage2-dir", default=None, help="Default: <stage1-dir>/../stage2.")
    parser.add_argument("--target", default="both", choices=["stage1", "stage2", "both"])
    parser.set_defaults(func=main_eval)


def build_argparser():
    parser = argparse.ArgumentParser(
        prog="btpc-train",
        description="Barlow Twins Stage 1/2 pipeline with anchor-free periodic filtering.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_stage1_parser(subparsers)
    build_stage2_parser(subparsers)
    build_eval_parser(subparsers)
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
