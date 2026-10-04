"""Labelled anchor utilities.

A small "anchor bank" of known-polarity SCSN waveforms is sampled and embedded
together with target waveforms at prediction time. Clusters are mapped to
physical up/down polarities by anchor purity (Hungarian matching between
clusters and anchor labels), which turns the Stage 2 A/B clusters into
seismological up/down classes.

The bank is built once from the SCSN training HDF5 (see docs/data.md) and
stored as an ``.npz`` file next to its CSV manifest.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import (
    DEFAULT_LABEL_KEY,
    DEFAULT_P_ARRIVAL_INDEX,
    DEFAULT_SNR_KEY,
    read_scsn_rows,
)
from .utils import checked_path, csv_text

POLARITY_NAMES = {
    0: "up",
    1: "down",
    2: "uncertain",
}

DEFAULT_ANCHOR_SNR_RANGE = (50.0, 1000.0)
DEFAULT_ANCHOR_POOL_PER_LABEL = 5000
DEFAULT_ANCHORS_PER_LABEL = 100
DEFAULT_ANCHOR_BUILD_SEED = 42
DEFAULT_ANCHOR_SAMPLE_SEED = 20260404


def polarity_name(label):
    return POLARITY_NAMES.get(int(label), f"label_{int(label)}")


def build_anchor_bank(
    bank_path,
    manifest_path=None,
    data_path=None,
    snr_range=DEFAULT_ANCHOR_SNR_RANGE,
    pool_per_label=DEFAULT_ANCHOR_POOL_PER_LABEL,
    seed=DEFAULT_ANCHOR_BUILD_SEED,
):
    """Sample a balanced pool of labelled SCSN anchors and save it as npz + CSV."""
    if not data_path:
        raise ValueError(
            "Building an anchor bank requires the SCSN training HDF5 path "
            "(data_path); see docs/data.md."
        )
    bank_path = str(bank_path)
    if manifest_path is None:
        manifest_path = os.path.splitext(bank_path)[0] + "_manifest.csv"
    bank_dir = os.path.dirname(os.path.abspath(bank_path)) or "."
    os.makedirs(bank_dir, exist_ok=True)

    with h5py.File(data_path, "r") as handle:
        if DEFAULT_LABEL_KEY not in handle or DEFAULT_SNR_KEY not in handle:
            available_keys = ", ".join(sorted(handle.keys()))
            raise KeyError(
                f"Anchor bank source {data_path} must contain "
                f"'{DEFAULT_LABEL_KEY}' and '{DEFAULT_SNR_KEY}'; available keys: {available_keys}"
            )
        labels_all = np.asarray(handle[DEFAULT_LABEL_KEY][:]).reshape(-1)
        snrs_all = np.asarray(handle[DEFAULT_SNR_KEY][:]).reshape(-1)

    low, high = snr_range
    rng = np.random.default_rng(seed)
    selected_indices = []
    for label in (0, 1):
        eligible = np.where(
            (labels_all == label) & (snrs_all >= low) & (snrs_all < high)
        )[0]
        if eligible.size < pool_per_label:
            raise ValueError(
                f"Not enough SCSN anchors for label={label} in SNR range "
                f"[{low}, {high}): need {pool_per_label}, found {eligible.size}"
            )
        chosen = np.sort(rng.choice(eligible, size=pool_per_label, replace=False))
        selected_indices.append(chosen)

    source_indices = np.concatenate(selected_indices, axis=0)
    raw_waveforms, labels, snrs, _ = read_scsn_rows(data_path, source_indices)

    anchor_record_ids = -(np.arange(len(labels), dtype=np.int64) + 1)
    anchor_keys = np.asarray(
        [f"anchor_scsn_{idx:06d}" for idx in range(len(labels))], dtype="U32"
    )

    np.savez_compressed(
        checked_path(bank_path),
        raw_waveforms=np.asarray(raw_waveforms, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        snrs=np.asarray(snrs, dtype=np.float32),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        anchor_record_ids=np.asarray(anchor_record_ids, dtype=np.int64),
        anchor_keys=anchor_keys,
        p_arrival_index=np.asarray(DEFAULT_P_ARRIVAL_INDEX, dtype=np.int64),
        snr_min=np.asarray(low, dtype=np.float32),
        snr_max=np.asarray(high, dtype=np.float32),
        pool_per_label=np.asarray(pool_per_label, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
    )

    manifest_rows = []
    for idx in range(len(labels)):
        manifest_rows.append(
            [
                int(anchor_record_ids[idx]),
                str(anchor_keys[idx]),
                int(source_indices[idx]),
                int(labels[idx]),
                polarity_name(labels[idx]),
                float(snrs[idx]),
            ]
        )
    manifest_header = [
        "anchor_record_id",
        "anchor_key",
        "source_index",
        "label",
        "label_name",
        "snr",
    ]
    Path(checked_path(manifest_path)).write_text(
        csv_text(manifest_header, manifest_rows), encoding="utf-8"
    )

    print(f"Saved anchor bank to: {bank_path}")
    print(f"Saved anchor manifest to: {manifest_path}")
    return bank_path


def ensure_anchor_bank(
    bank_path,
    manifest_path=None,
    data_path=None,
    snr_range=DEFAULT_ANCHOR_SNR_RANGE,
    pool_per_label=DEFAULT_ANCHOR_POOL_PER_LABEL,
    seed=DEFAULT_ANCHOR_BUILD_SEED,
):
    """Return ``bank_path``, building the bank first if it does not exist yet."""
    if bank_path is None:
        raise ValueError(
            "No anchor bank given. Pass --anchor-bank-path; build a bank with "
            "btpc.anchor_utils.build_anchor_bank (see docs/data.md)."
        )
    if not os.path.exists(bank_path):
        build_anchor_bank(
            bank_path=bank_path,
            manifest_path=manifest_path,
            data_path=data_path,
            snr_range=snr_range,
            pool_per_label=pool_per_label,
            seed=seed,
        )
    return bank_path


def load_anchor_bank(bank_path):
    with np.load(bank_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def anchor_event_ids(
    bank_path,
    source_path,
    n_per_label=DEFAULT_ANCHORS_PER_LABEL,
    seed=DEFAULT_ANCHOR_SAMPLE_SEED,
):
    """Event ids of the reference subset drawn from the bank at prediction time.

    Replays ``sample_anchor_subset`` (default 100 per label, seed 20260404),
    then looks the bank rows up in the SCSN source file that
    ``source_indices`` point into. Used to keep reference events out of the
    held-out splits (ticket 01, R1.4).
    """
    bank = load_anchor_bank(checked_path(bank_path))
    subset = sample_anchor_subset(bank, n_per_label=n_per_label, seed=seed)
    source_indices = np.sort(np.asarray(subset["source_indices"], dtype=np.int64))
    with h5py.File(checked_path(source_path), "r") as handle:
        return np.unique(handle["evids"][source_indices])


def anchor_pool_event_ids(bank_path, source_path):
    """Event ids of every row in the anchor bank pool (not just the drawn subset)."""
    bank = load_anchor_bank(checked_path(bank_path))
    source_indices = np.sort(np.asarray(bank["source_indices"], dtype=np.int64))
    with h5py.File(checked_path(source_path), "r") as handle:
        return handle["evids"][source_indices]


def sample_anchor_subset(
    bank,
    n_per_label=DEFAULT_ANCHORS_PER_LABEL,
    seed=DEFAULT_ANCHOR_SAMPLE_SEED,
):
    labels = np.asarray(bank["labels"], dtype=np.int64)
    rng = np.random.default_rng(seed)

    selected = []
    for label in (0, 1):
        eligible = np.where(labels == label)[0]
        if eligible.size < n_per_label:
            raise ValueError(
                f"Anchor bank does not have enough samples for label={label}: "
                f"need {n_per_label}, found {eligible.size}"
            )
        selected.append(rng.choice(eligible, size=n_per_label, replace=False))

    chosen = np.concatenate(selected, axis=0)
    chosen = chosen[rng.permutation(chosen.size)]

    subset = {
        "raw_waveforms": np.asarray(bank["raw_waveforms"][chosen], dtype=np.float32),
        "labels": np.asarray(bank["labels"][chosen], dtype=np.int64),
        "snrs": np.asarray(bank["snrs"][chosen], dtype=np.float32),
        "source_indices": np.asarray(bank["source_indices"][chosen], dtype=np.int64),
        "anchor_record_ids": np.asarray(bank["anchor_record_ids"][chosen], dtype=np.int64),
        "anchor_keys": np.asarray(bank["anchor_keys"][chosen]),
        "p_arrival_index": int(np.asarray(bank["p_arrival_index"]).item()),
        "seed": int(seed),
    }
    return subset


def max_normalize_waveforms(waveforms):
    waveforms = np.asarray(waveforms, dtype=np.float32)
    reduce_axes = tuple(range(1, waveforms.ndim))
    max_vals = np.max(np.abs(waveforms), axis=reduce_axes, keepdims=True)
    max_vals[max_vals < 1e-12] = 1.0
    return waveforms / max_vals


def crop_waveforms_around_index(waveforms, center_index, size, shift=0):
    waveforms = np.asarray(waveforms, dtype=np.float32)
    start = int(center_index) - int(size) // 2 + int(shift)
    end = start + int(size)
    if start < 0 or end > waveforms.shape[1]:
        raise ValueError(
            f"Invalid crop window [{start}, {end}) for waveform length {waveforms.shape[1]}"
        )
    return waveforms[:, start:end]


def prepare_anchor_waveforms_for_ridge(
    raw_waveforms,
    resize,
    shift,
    p_arrival_index=DEFAULT_P_ARRIVAL_INDEX,
):
    """Crop anchors to the model window and max-normalize them."""
    cropped = crop_waveforms_around_index(
        raw_waveforms,
        center_index=p_arrival_index,
        size=resize,
        shift=shift,
    )
    return max_normalize_waveforms(cropped)


def fit_anchor_cluster_mapping(cluster_labels, is_anchor, known_labels) -> Tuple[Dict, Dict]:
    """Map cluster IDs to up/down by Hungarian matching on anchor labels."""
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    is_anchor = np.asarray(is_anchor).astype(bool)
    known_labels = np.asarray(known_labels, dtype=np.int64)

    anchor_mask = is_anchor & np.isin(known_labels, [0, 1])
    if not np.any(anchor_mask):
        raise ValueError("No valid anchor labels found while fitting cluster mapping.")

    anchor_clusters = cluster_labels[anchor_mask]
    anchor_labels = known_labels[anchor_mask]

    cluster_ids = sorted(np.unique(cluster_labels).tolist())
    label_ids = [0, 1]
    confusion = np.zeros((len(cluster_ids), len(label_ids)), dtype=np.int64)

    for row_idx, cluster_id in enumerate(cluster_ids):
        cluster_mask = anchor_clusters == cluster_id
        for col_idx, label_id in enumerate(label_ids):
            confusion[row_idx, col_idx] = int(np.sum(anchor_labels[cluster_mask] == label_id))

    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {cluster_ids[row]: label_ids[col] for row, col in zip(row_ind, col_ind)}

    stats = {}
    for row_idx, cluster_id in enumerate(cluster_ids):
        up_count = int(confusion[row_idx, 0])
        down_count = int(confusion[row_idx, 1])
        total = up_count + down_count
        mapped_label = mapping.get(cluster_id, -1)
        mapped_count = 0
        if mapped_label in (0, 1):
            mapped_count = up_count if mapped_label == 0 else down_count
        purity = float(mapped_count / total) if total > 0 else 0.0
        stats[int(cluster_id)] = {
            "mapped_label": int(mapped_label),
            "up_count": up_count,
            "down_count": down_count,
            "anchor_total": total,
            "purity": purity,
        }

    return mapping, stats


def apply_cluster_mapping(cluster_labels, mapping, default_label=-1):
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    mapped = np.full(cluster_labels.shape, int(default_label), dtype=np.int64)
    for cluster_id, label_id in mapping.items():
        mapped[cluster_labels == int(cluster_id)] = int(label_id)
    return mapped
