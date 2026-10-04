"""Prediction-time anchoring and uncertain rejection.

The Stage 2 model outputs cluster A/B classes. This script maps them to
physical up/down polarities with a small labelled SCSN anchor bank, runs the
mapping either on unused rows of the training file (``--target-source
train_unused``) or on a Ridgecrest waveform file (``--target-source ridge``),
and rejects unstable predictions (low TTA vote consistency or low margin).
"""

import argparse
import csv
import io
import os
from pathlib import Path
from typing import Dict, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset

from .anchor_utils import (
    apply_cluster_mapping,
    ensure_anchor_bank,
    fit_anchor_cluster_mapping,
    load_anchor_bank,
    polarity_name,
    prepare_anchor_waveforms_for_ridge,
    sample_anchor_subset,
)
from .data import (
    DEFAULT_LABEL_KEY,
    DEFAULT_SNR_KEY,
    DEFAULT_WAVEFORM_KEY,
    SeismicPolarityDataset,
    read_scsn_rows,
    weak_eval_augment_waveforms,
)
from .models import build_stage2_classifier
from .train import get_data_selection_seed
from .utils import (
    checked_path,
    compute_cluster_margin,
    dedim,
    device,
    load_checkpoint,
    preprocess_cluster_features,
    resolve_data_path,
    save_json,
)


class RidgePredictDataset(Dataset):
    def __init__(
        self,
        waveforms,
        snrs,
        record_ids,
        labels=None,
        source_indices=None,
        source_name="ridge",
    ):
        self.waveforms = np.asarray(waveforms, dtype=np.float32)
        self.snrs = np.asarray(snrs, dtype=np.float32)
        self.record_ids = np.asarray(record_ids)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
        if source_indices is None:
            source_indices = np.arange(len(self.waveforms), dtype=np.int64)
        self.source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        self.source_name = str(source_name)

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.waveforms[idx]),
            torch.tensor(float(self.snrs[idx]), dtype=torch.float32),
            self.record_ids[idx],
        )


def load_data_ridge(data_path: str, resize: int, shift: int, snr_min: float, snr_max: float, limit: int = 0):
    data_path = resolve_data_path(data_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Ridgecrest data file not found: {data_path}. "
            "Pass --data-path pointing to the consensus_waveforms HDF5 file."
        )
    with h5py.File(data_path, "r") as handle:
        if "phasenet" not in handle:
            available_keys = ", ".join(sorted(handle.keys()))
            raise KeyError(
                f"Missing group 'phasenet' in {data_path}. Available groups: {available_keys}"
            )
        group = handle["phasenet"]
        waveforms = group["waveforms"][:]
        snrs = group["snr"][:]
        record_ids = group["record_id"][:]

    snrs = np.asarray(snrs).reshape(-1)
    mask = (snrs >= snr_min) & (snrs < snr_max)
    valid_idx = np.where(mask)[0]
    if limit and valid_idx.size > limit:
        valid_idx = valid_idx[:limit]

    waveforms = np.asarray(waveforms[valid_idx], dtype=np.float32)
    snrs = snrs[valid_idx]
    record_ids = record_ids[valid_idx]

    waveforms = signal.detrend(waveforms, axis=1).astype(np.float32, copy=False)

    sampling_rate = 100.0
    p_arrival = int(10 * sampling_rate)
    start_idx = p_arrival - resize // 2 + shift
    end_idx = p_arrival + resize // 2 + shift
    waveforms = waveforms[:, start_idx:end_idx]

    reduce_axes = tuple(range(1, waveforms.ndim))
    max_vals = np.max(np.abs(waveforms), axis=reduce_axes, keepdims=True)
    max_vals[max_vals < 1e-12] = 1.0
    waveforms = waveforms / max_vals
    return RidgePredictDataset(waveforms, snrs, record_ids, source_indices=valid_idx, source_name="ridge")


def _stage1_data_settings(stage1_config: Dict, train_data_path_override: Optional[str] = None) -> Dict:
    data_path = train_data_path_override or stage1_config.get(
        "train_data_path", stage1_config.get("data_path")
    )
    if not data_path:
        raise ValueError(
            "stage1_config.json does not record a training data path; "
            "pass --train-data-path explicitly."
        )
    return {
        "data_path": resolve_data_path(data_path),
        "waveform_key": stage1_config.get("data_waveform_key", DEFAULT_WAVEFORM_KEY),
        "label_key": stage1_config.get("data_label_key", DEFAULT_LABEL_KEY),
        "snr_key": stage1_config.get("data_snr_key", DEFAULT_SNR_KEY),
    }


def get_stage1_selection_pools(
    stage1_config: Dict, train_data_path_override: Optional[str] = None
):
    """Reproduce the Stage 1 sample selection from its saved config.

    Returns ``(training_source_indices, candidate_pool)`` where the candidate
    pool is the pre-filter seeded 2x draw (empty bino filtering not applied).
    """
    data_path = train_data_path_override or stage1_config.get(
        "train_data_path", stage1_config.get("data_path")
    )
    if not data_path:
        raise ValueError(
            "stage1_config.json does not record a training data path; "
            "pass train_data_path_override explicitly."
        )
    data_path = resolve_data_path(data_path)
    label_key = stage1_config.get("data_label_key", DEFAULT_LABEL_KEY)
    snr_key = stage1_config.get("data_snr_key", DEFAULT_SNR_KEY)
    low, high = stage1_config.get("snr_range", [0.0, 1000.0])
    n_select = int(stage1_config.get("num_used", 10000))
    selection_seed = get_data_selection_seed(stage1_config)
    bino = bool(stage1_config.get("bino", True))

    with h5py.File(data_path, "r") as handle:
        snr_all = np.asarray(handle[snr_key][:]).reshape(-1)
        labels_all = np.asarray(handle[label_key][:]).reshape(-1)

    eligible = np.where((snr_all >= float(low)) & (snr_all < float(high)))[0]
    if eligible.size == 0:
        raise ValueError(f"No train samples in stage1 snr range [{low}, {high}).")

    if n_select > 0:
        if eligible.size < n_select:
            raise ValueError(
                f"Only {eligible.size} train samples are in stage1 snr range [{low}, {high}), "
                f"cannot reproduce n_select={n_select}."
            )
        rng = np.random.default_rng(selection_seed)
        candidate_count = min(eligible.size, n_select * 2)
        candidate_pool = rng.choice(eligible, size=candidate_count, replace=False)
    else:
        candidate_pool = eligible

    selected = candidate_pool
    if bino:
        selected = selected[labels_all[selected] != 2]

    if n_select > 0:
        if selected.size < n_select:
            raise ValueError(
                f"Only {selected.size} samples remain after reproducing stage1 filtering, "
                f"cannot reproduce n_select={n_select}."
            )
        selected = selected[:n_select]

    return np.asarray(selected, dtype=np.int64), np.asarray(candidate_pool, dtype=np.int64)


def get_stage1_training_source_indices(
    stage1_config: Dict, train_data_path_override: Optional[str] = None
) -> np.ndarray:
    """Reproduce the exact sample selection used during Stage 1 training."""
    selected, _ = get_stage1_selection_pools(stage1_config, train_data_path_override)
    return selected


def load_data_train_unused(
    stage1_config: Dict,
    snr_min: float,
    snr_max: float,
    limit: int = 0,
    sample_seed: int = 20260530,
    train_data_path_override: Optional[str] = None,
):
    data_settings = _stage1_data_settings(stage1_config, train_data_path_override)
    trained_indices = get_stage1_training_source_indices(stage1_config, train_data_path_override)
    bino = bool(stage1_config.get("bino", True))

    with h5py.File(data_settings["data_path"], "r") as handle:
        snr_all = np.asarray(handle[data_settings["snr_key"]][:]).reshape(-1)
        labels_all = np.asarray(handle[data_settings["label_key"]][:]).reshape(-1)

    target_mask = (snr_all >= float(snr_min)) & (snr_all < float(snr_max))
    if bino:
        target_mask &= labels_all != 2

    candidate_indices = np.where(target_mask)[0]
    unused_indices = np.setdiff1d(candidate_indices, trained_indices, assume_unique=False)
    if unused_indices.size == 0:
        raise ValueError("No unused train samples remain after excluding stage1 training samples.")

    requested_limit = int(limit) if limit and limit > 0 else 0
    if requested_limit and unused_indices.size > requested_limit:
        rng = np.random.default_rng(sample_seed)
        source_indices = rng.choice(unused_indices, size=requested_limit, replace=False)
    else:
        source_indices = unused_indices

    raw_waveforms, labels, snrs, _ = read_scsn_rows(
        data_settings["data_path"],
        source_indices,
        waveform_key=data_settings["waveform_key"],
        label_key=data_settings["label_key"],
        snr_key=data_settings["snr_key"],
    )
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    snrs = np.asarray(snrs, dtype=np.float32).reshape(-1)

    base_dataset = SeismicPolarityDataset(
        waveforms=raw_waveforms,
        polarities=labels,
        snrs=snrs,
        p_arrival_times=np.full(len(raw_waveforms), 300, dtype=np.int64),
        sampling_rate=100,
        shift=int(stage1_config["shift"]),
        p_window=int(stage1_config["resize"]) / 100,
        apply_augmentation=False,
        aug_shift=int(stage1_config.get("aug_shift", 1)),
        aug_noise_std_range=stage1_config.get("aug_noise_std_range", (0.05, 0.2)),
        aug_scale_range=stage1_config.get("aug_scale_range", (0.8, 1.2)),
        norm_mod=str(stage1_config["norm_mod"]),
    )
    clean_waveforms = base_dataset.get_clean_waveforms(np.arange(len(base_dataset), dtype=np.int64))
    record_ids = np.asarray([f"train_unused_{int(idx)}" for idx in source_indices], dtype=object)

    metadata = {
        "data_path": data_settings["data_path"],
        "waveform_key": data_settings["waveform_key"],
        "label_key": data_settings["label_key"],
        "snr_key": data_settings["snr_key"],
        "trained_sample_count": int(trained_indices.size),
        "unused_candidate_count": int(unused_indices.size),
        "selected_sample_count": int(source_indices.size),
        "target_sample_seed": int(sample_seed),
    }
    return RidgePredictDataset(
        clean_waveforms,
        snrs,
        record_ids,
        labels=labels,
        source_indices=source_indices,
        source_name="train_unused",
    ), metadata


def load_stage2_bundle(checkpoint_path: str):
    checkpoint_path = resolve_data_path(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {checkpoint_path}")
    bundle = load_checkpoint(checkpoint_path, map_location=device)
    stage1_config = bundle["stage1_config"]

    classifier = build_stage2_classifier(
        base_channels=int(stage1_config["base_cha"]),
        head_hidden_dim=int(bundle.get("head_hidden_dim", 0)),
    )
    classifier.encoder.load_state_dict(bundle["encoder_state_dict"])
    classifier.head.load_state_dict(bundle["head_state_dict"])
    classifier.to(device)
    classifier.eval()
    return classifier, bundle


def run_classifier_logits(model, waveforms: np.ndarray, batch_size: int):
    waves = np.asarray(waveforms, dtype=np.float32)
    if waves.ndim == 2:
        waves = waves[:, None, :]

    logits = []
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, waves.shape[0], batch_size):
            batch = torch.from_numpy(waves[start : start + batch_size]).to(device)
            feat = model.encode(batch)
            logit = model.head(feat)
            embeddings.append(feat.cpu().numpy())
            logits.append(logit.cpu().numpy())
    return np.vstack(logits), np.vstack(embeddings)


def normalize_features(raw_features: np.ndarray, feature_mean: np.ndarray, feature_scale: np.ndarray):
    features, _, _ = preprocess_cluster_features(
        raw_features,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    return features


def softmax_numpy(logits: np.ndarray):
    logits = np.asarray(logits, dtype=float)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)


def compute_prediction_metrics(
    model,
    clean_waveforms: np.ndarray,
    cluster_centers: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    n_tta_views: int,
    tta_max_shift: int,
    batch_size: int,
    tta_scale_jitter: float = 0.02,
    tta_noise_std: float = 0.005,
):
    repeated_classes = []
    repeated_confidence = []
    repeated_margin = []
    repeated_center_margin = []
    repeated_features = []

    total_views = max(int(n_tta_views), 1)
    for view_idx in range(total_views):
        if view_idx == 0:
            waves = clean_waveforms
        else:
            waves = weak_eval_augment_waveforms(
                clean_waveforms,
                max_shift=tta_max_shift,
                scale_jitter=float(tta_scale_jitter),
                noise_std=float(tta_noise_std),
            )
        logits, raw_embeddings = run_classifier_logits(model, waves, batch_size=batch_size)
        probs = softmax_numpy(logits)
        classes = probs.argmax(axis=1)
        margin = np.abs(probs[:, 0] - probs[:, 1])
        norm_features = normalize_features(raw_embeddings, feature_mean, feature_scale)
        center_margin = compute_cluster_margin(norm_features, cluster_centers)

        repeated_classes.append(classes)
        repeated_confidence.append(probs.max(axis=1))
        repeated_margin.append(margin)
        repeated_center_margin.append(center_margin)
        repeated_features.append(norm_features)

    repeated_classes = np.stack(repeated_classes, axis=0)
    repeated_confidence = np.stack(repeated_confidence, axis=0)
    repeated_margin = np.stack(repeated_margin, axis=0)
    repeated_center_margin = np.stack(repeated_center_margin, axis=0)
    repeated_features = np.stack(repeated_features, axis=0)

    modal_class = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=2).argmax(), 0, repeated_classes
    )
    vote_consistency = (repeated_classes == modal_class[None, :]).mean(axis=0)
    mean_confidence = repeated_confidence.mean(axis=0)
    mean_margin = repeated_margin.mean(axis=0)
    mean_center_margin = repeated_center_margin.mean(axis=0)
    feature_drift = np.linalg.norm(repeated_features - repeated_features[0:1], axis=2).mean(axis=0)
    return {
        "modal_class": modal_class.astype(np.int64),
        "vote_consistency": vote_consistency,
        "mean_confidence": mean_confidence,
        "mean_margin": mean_margin,
        "mean_center_margin": mean_center_margin,
        "embedding_drift": feature_drift,
    }


def save_prediction_summary(save_dir: str, payload: Dict):
    save_json(os.path.join(save_dir, "predict_summary.json"), payload)


def plot_prediction_clusters(features: np.ndarray, labels: np.ndarray, save_dir: str):
    emb = dedim(features, ver="tsne")
    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="viridis", s=5, alpha=0.4)
    plt.title("Prediction-time cluster classes")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_clusters_tsne.png"), dpi=200)
    plt.close()


def plot_prediction_labels(features: np.ndarray, labels: np.ndarray, save_dir: str):
    emb = dedim(features, ver="tsne")
    plt.figure(figsize=(8, 6))

    plot_specs = [
        (0, "up", "#1f77b4"),
        (1, "down", "#d62728"),
        (2, "uncertain/reject", "#7f7f7f"),
    ]
    for label_value, label_name, color in plot_specs:
        mask = labels == label_value
        if not np.any(mask):
            continue
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=0.5,
            c=color,
            label=f"{label_name} (n={int(mask.sum())})",
        )

    plt.title("Prediction-time up/down/reject labels")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(markerscale=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_up_down_reject_tsne.png"), dpi=200)
    plt.close()


def compute_accuracy_summary(
    true_labels: np.ndarray, pred_labels: np.ndarray, final_labels: np.ndarray
) -> Dict:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    final_labels = np.asarray(final_labels, dtype=np.int64)

    binary_mask = np.isin(true_labels, [0, 1])
    accepted_mask = binary_mask & (final_labels != 2)
    summary = {
        "binary_count": int(np.sum(binary_mask)),
        "accepted_count": int(np.sum(accepted_mask)),
        "reject_count": int(np.sum(binary_mask & (final_labels == 2))),
        "accept_rate": float(np.mean(final_labels[binary_mask] != 2)) if np.any(binary_mask) else None,
        "accuracy_before_reject": (
            float(np.mean(pred_labels[binary_mask] == true_labels[binary_mask]))
            if np.any(binary_mask)
            else None
        ),
        "accuracy_after_reject_reject_as_wrong": (
            float(np.mean(final_labels[binary_mask] == true_labels[binary_mask]))
            if np.any(binary_mask)
            else None
        ),
        "accuracy_on_accepted_only": (
            float(np.mean(pred_labels[accepted_mask] == true_labels[accepted_mask]))
            if np.any(accepted_mask)
            else None
        ),
        "per_true_label": {},
    }

    for label in (0, 1):
        label_mask = true_labels == label
        label_accepted_mask = label_mask & (final_labels != 2)
        summary["per_true_label"][polarity_name(label)] = {
            "support": int(np.sum(label_mask)),
            "accepted_count": int(np.sum(label_accepted_mask)),
            "reject_count": int(np.sum(label_mask & (final_labels == 2))),
            "accept_rate": (
                float(np.mean(final_labels[label_mask] != 2)) if np.any(label_mask) else None
            ),
            "accuracy_before_reject": (
                float(np.mean(pred_labels[label_mask] == true_labels[label_mask]))
                if np.any(label_mask)
                else None
            ),
            "accuracy_after_reject_reject_as_wrong": (
                float(np.mean(final_labels[label_mask] == true_labels[label_mask]))
                if np.any(label_mask)
                else None
            ),
            "accuracy_on_accepted_only": (
                float(np.mean(pred_labels[label_accepted_mask] == true_labels[label_accepted_mask]))
                if np.any(label_accepted_mask)
                else None
            ),
        }

    return summary


def format_metric(value) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def main_predict(args):
    model, bundle = load_stage2_bundle(args.stage2_checkpoint)
    stage1_config = bundle["stage1_config"]
    resize = int(stage1_config["resize"])
    shift = int(stage1_config["shift"])
    batch_size = int(stage1_config["batch_size"])

    default_predict_subdir = (
        "predict_train_unused_out" if args.target_source == "train_unused" else "predict_out"
    )
    save_dir = args.save_dir or os.path.join(
        os.path.dirname(str(args.stage2_checkpoint)), default_predict_subdir
    )
    os.makedirs(save_dir, exist_ok=True)

    target_metadata = {}
    if args.target_source == "train_unused":
        target_dataset, target_metadata = load_data_train_unused(
            stage1_config=stage1_config,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
            limit=args.max_samples,
            sample_seed=args.target_sample_seed,
            train_data_path_override=args.train_data_path,
        )
    else:
        if not args.data_path:
            raise ValueError(
                "--data-path is required when --target-source ridge "
                "(the Ridgecrest consensus HDF5 file)."
            )
        target_dataset = load_data_ridge(
            data_path=args.data_path,
            resize=resize,
            shift=shift,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
            limit=args.max_samples,
        )
        target_metadata = {
            "data_path": resolve_data_path(args.data_path),
            "selected_sample_count": int(len(target_dataset)),
        }
    clean_waveforms = np.asarray(target_dataset.waveforms, dtype=np.float32)

    bank_path = ensure_anchor_bank(bank_path=args.anchor_bank_path)
    bank = load_anchor_bank(bank_path)
    anchor_subset = sample_anchor_subset(
        bank, n_per_label=args.anchors_per_label, seed=args.anchor_sample_seed
    )
    anchor_waveforms = prepare_anchor_waveforms_for_ridge(
        anchor_subset["raw_waveforms"],
        resize=resize,
        shift=shift,
        p_arrival_index=int(anchor_subset["p_arrival_index"]),
    )

    cluster_centers = np.asarray(bundle["cluster_centers"], dtype=np.float32)
    feature_mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(bundle["feature_scale"], dtype=np.float32)

    anchor_metrics = compute_prediction_metrics(
        model=model,
        clean_waveforms=anchor_waveforms,
        cluster_centers=cluster_centers,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        n_tta_views=max(args.n_tta_views, 1),
        tta_max_shift=args.tta_max_shift,
        batch_size=batch_size,
        tta_scale_jitter=args.tta_scale_jitter,
        tta_noise_std=(
            args.tta_noise_std if args.anchor_tta_noise_std is None else args.anchor_tta_noise_std
        ),
    )
    mapping, cluster_stats = fit_anchor_cluster_mapping(
        cluster_labels=anchor_metrics["modal_class"],
        is_anchor=np.ones_like(anchor_metrics["modal_class"], dtype=np.int64),
        known_labels=np.asarray(anchor_subset["labels"], dtype=np.int64),
    )
    print("Prediction-time anchor mapping:")
    for cluster_id in sorted(cluster_stats):
        stat = cluster_stats[cluster_id]
        print(
            f"  class {cluster_id} -> {polarity_name(stat['mapped_label'])} | "
            f"up={stat['up_count']} | down={stat['down_count']} | purity={stat['purity']:.4f}"
        )

    target_metrics = compute_prediction_metrics(
        model=model,
        clean_waveforms=clean_waveforms,
        cluster_centers=cluster_centers,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        n_tta_views=max(args.n_tta_views, 1),
        tta_max_shift=args.tta_max_shift,
        batch_size=batch_size,
        tta_scale_jitter=args.tta_scale_jitter,
        tta_noise_std=args.tta_noise_std,
    )
    pred_polarity = apply_cluster_mapping(target_metrics["modal_class"], mapping)

    reject_mask = (
        (target_metrics["vote_consistency"] < args.predict_vote_threshold)
        | (target_metrics["mean_margin"] < args.predict_margin_threshold)
    )
    final_label = pred_polarity.copy()
    final_label[reject_mask] = 2

    _, raw_embeddings = run_classifier_logits(model, clean_waveforms, batch_size=batch_size)
    norm_features = normalize_features(raw_embeddings, feature_mean, feature_scale)
    plot_prediction_clusters(norm_features, target_metrics["modal_class"], save_dir)
    plot_prediction_labels(norm_features, final_label, save_dir)

    csv_path = os.path.join(save_dir, "predictions_anchor_mapped.csv")
    header = [
        "record_id",
        "target_source",
        "source_index",
        "snr",
        "true_label",
        "true_label_name",
        "pred_cluster_class",
        "pred_polarity",
        "pred_polarity_name",
        "vote_consistency",
        "mean_confidence",
        "mean_margin",
        "mean_center_margin",
        "embedding_drift",
        "is_reject",
        "final_label",
        "final_label_name",
    ]
    rows = []
    for idx in range(len(target_dataset)):
        record_id = target_dataset.record_ids[idx]
        if isinstance(record_id, bytes):
            record_id = record_id.decode("utf-8", errors="ignore")
        true_label = "" if target_dataset.labels is None else int(target_dataset.labels[idx])
        true_label_name = "" if target_dataset.labels is None else polarity_name(true_label)
        rows.append(
            [
                record_id,
                target_dataset.source_name,
                int(target_dataset.source_indices[idx]),
                float(target_dataset.snrs[idx]),
                true_label,
                true_label_name,
                int(target_metrics["modal_class"][idx]),
                int(pred_polarity[idx]),
                polarity_name(pred_polarity[idx]),
                float(target_metrics["vote_consistency"][idx]),
                float(target_metrics["mean_confidence"][idx]),
                float(target_metrics["mean_margin"][idx]),
                float(target_metrics["mean_center_margin"][idx]),
                float(target_metrics["embedding_drift"][idx]),
                int(bool(reject_mask[idx])),
                int(final_label[idx]),
                polarity_name(final_label[idx]),
            ]
        )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    Path(checked_path(csv_path)).write_text(buffer.getvalue(), encoding="utf-8")

    target_counts = {
        "up": int(np.sum(final_label == 0)),
        "down": int(np.sum(final_label == 1)),
        "uncertain": int(np.sum(final_label == 2)),
    }
    true_label_summary = None
    accuracy_summary = None
    if target_dataset.labels is not None:
        true_labels = np.asarray(target_dataset.labels, dtype=np.int64)
        true_label_summary = {
            polarity_name(label): int(np.sum(true_labels == label))
            for label in sorted(np.unique(true_labels).tolist())
        }
        comparable_mask = np.isin(true_labels, [0, 1]) & (final_label != 2)
        true_label_summary["non_reject_comparable_count"] = int(np.sum(comparable_mask))
        if np.any(comparable_mask):
            true_label_summary["non_reject_accuracy"] = float(
                np.mean(final_label[comparable_mask] == true_labels[comparable_mask])
            )
        else:
            true_label_summary["non_reject_accuracy"] = None
        accuracy_summary = compute_accuracy_summary(
            true_labels=true_labels,
            pred_labels=pred_polarity,
            final_labels=final_label,
        )

    save_prediction_summary(
        save_dir,
        {
            "stage2_checkpoint": args.stage2_checkpoint,
            "save_dir": save_dir,
            "csv_path": csv_path,
            "target_source": args.target_source,
            "target_metadata": target_metadata,
            "max_samples": int(args.max_samples),
            "snr_range": [float(args.snr_min), float(args.snr_max)],
            "anchor_bank_path": bank_path,
            "anchors_per_label": int(args.anchors_per_label),
            "anchor_sample_seed": int(args.anchor_sample_seed),
            "cluster_mapping": {int(k): int(v) for k, v in mapping.items()},
            "cluster_stats": cluster_stats,
            "thresholds": {
                "predict_vote_threshold": float(args.predict_vote_threshold),
                "predict_margin_threshold": float(args.predict_margin_threshold),
            },
            "label_counts": target_counts,
            "true_label_summary": true_label_summary,
            "accuracy_summary": accuracy_summary,
            "note": (
                "Anchor data is only used at prediction time to map cluster A/B outputs "
                "to up/down and reject unstable predictions."
            ),
        },
    )
    print(f"Saved prediction csv to: {csv_path}")
    print(
        f"Prediction label counts | up={target_counts['up']} | down={target_counts['down']} | "
        f"uncertain={target_counts['uncertain']}"
    )
    if accuracy_summary is not None:
        up_acc = accuracy_summary["per_true_label"]["up"]["accuracy_before_reject"]
        down_acc = accuracy_summary["per_true_label"]["down"]["accuracy_before_reject"]
        up_acc_final = accuracy_summary["per_true_label"]["up"]["accuracy_after_reject_reject_as_wrong"]
        down_acc_final = accuracy_summary["per_true_label"]["down"]["accuracy_after_reject_reject_as_wrong"]
        print(
            "Accuracy before reject | "
            f"overall={format_metric(accuracy_summary['accuracy_before_reject'])} | "
            f"up={format_metric(up_acc)} | down={format_metric(down_acc)}"
        )
        print(
            "Accuracy after reject (reject as wrong) | "
            f"overall={format_metric(accuracy_summary['accuracy_after_reject_reject_as_wrong'])} | "
            f"up={format_metric(up_acc_final)} | down={format_metric(down_acc_final)}"
        )
        if accuracy_summary["accuracy_on_accepted_only"] is not None:
            up_acc_accepted = accuracy_summary["per_true_label"]["up"]["accuracy_on_accepted_only"]
            down_acc_accepted = accuracy_summary["per_true_label"]["down"]["accuracy_on_accepted_only"]
            print(
                "Accuracy on accepted only | "
                f"overall={format_metric(accuracy_summary['accuracy_on_accepted_only'])} | "
                f"up={format_metric(up_acc_accepted)} | down={format_metric(down_acc_accepted)}"
            )


def build_argparser():
    parser = argparse.ArgumentParser(
        prog="btpc-predict",
        description="Stage 3 prediction-time anchoring and uncertain rejection.",
    )
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument(
        "--target-source",
        choices=["ridge", "train_unused"],
        default="ridge",
        help=(
            "ridge: a Ridgecrest consensus HDF5; train_unused: rows of the "
            "training file not used by Stage 1."
        ),
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Ridgecrest consensus HDF5 (required with --target-source ridge).",
    )
    parser.add_argument(
        "--train-data-path",
        default=None,
        help="Override the train HDF5 path saved in stage1_config; only used with --target-source train_unused.",
    )
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=1000.0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means predict all candidates.")
    parser.add_argument("--target-sample-seed", type=int, default=20260530)
    parser.add_argument(
        "--anchor-bank-path",
        default=None,
        help=(
            "Anchor bank npz file (required). Build one with "
            "btpc.anchor_utils.build_anchor_bank; see docs/data.md."
        ),
    )
    parser.add_argument("--anchors-per-label", type=int, default=100)
    parser.add_argument("--anchor-sample-seed", type=int, default=20260404)
    parser.add_argument("--n-tta-views", type=int, default=4)
    parser.add_argument("--tta-max-shift", type=int, default=2)
    parser.add_argument("--tta-scale-jitter", type=float, default=0.02)
    parser.add_argument("--tta-noise-std", type=float, default=0.005)
    parser.add_argument(
        "--anchor-tta-noise-std",
        type=float,
        default=None,
        help="Separate TTA noise std for the anchor mapping pass (default: same as --tta-noise-std).",
    )
    parser.add_argument("--predict-vote-threshold", type=float, default=0.8)
    parser.add_argument("--predict-margin-threshold", type=float, default=0.15)
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    main_predict(args)


if __name__ == "__main__":
    main()
