"""Shared helpers: device selection, JSON IO, path resolution, index runs,
label-matching metrics, clustering/feature utilities and plotting."""

import csv
import io
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
import sklearn.mixture
import sklearn.preprocessing
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFUSION_FIGSIZE = (8.5, 6.8)
CONFUSION_TITLE_FONTSIZE = 20
CONFUSION_LABEL_FONTSIZE = 18
CONFUSION_TICK_FONTSIZE = 18
CONFUSION_CELL_FONTSIZE = 18
CONFUSION_CBAR_FONTSIZE = 18


def checked_path(path) -> str:
    """Normalize a user-supplied file path and refuse '..' escape segments.

    Every file-writing call in the package funnels its target through this
    helper so that outputs always land on a fully resolved path.
    """
    resolved = os.path.realpath(os.path.expanduser(str(path)))
    if ".." in resolved.split(os.sep):
        raise ValueError(f"Refusing unsafe output path: {path}")
    return resolved


def maybe_mkdir(path: str):
    os.makedirs(checked_path(path), exist_ok=True)


def checked_input_path(data_path) -> str:
    """Resolve a user-supplied input path, refuse traversal, and verify it exists."""
    resolved = checked_path(data_path)
    if not Path(resolved).is_file():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    return resolved


def checked_output_path(output_dir, filename: str) -> str:
    """Resolve ``filename`` inside ``output_dir``, refusing any upward escape."""
    directory = Path(checked_path(output_dir))
    candidate = (directory / filename).resolve()
    if ".." in candidate.parts or not candidate.is_relative_to(directory):
        raise ValueError(f"Output path escapes {directory}: {candidate}")
    return str(candidate)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(path: str, payload: Dict):
    text = json.dumps(json_ready(payload), indent=2, ensure_ascii=False)
    Path(checked_path(path)).write_text(text, encoding="utf-8")


def load_checkpoint(path, map_location=None):
    """Load a checkpoint with ``weights_only=True`` (no arbitrary pickle execution).

    Stage 2 bundles carry plain numpy arrays next to the tensors; numpy's
    dtype/array globals are registered as safe because they are data, not code.
    """
    safe_globals = [
        np.ndarray,
        np.dtype,
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]
    try:
        from numpy._core.multiarray import _reconstruct, scalar as numpy_scalar
    except ImportError:  # numpy 1.x layout
        from numpy.core.multiarray import _reconstruct, scalar as numpy_scalar
    safe_globals.extend([_reconstruct, numpy_scalar])
    try:
        from numpy.dtypes import (
            BoolDType,
            Complex64DType,
            Complex128DType,
            Float16DType,
            Float32DType,
            Float64DType,
            Int8DType,
            Int16DType,
            Int32DType,
            Int64DType,
        )
        safe_globals.extend(
            [
                BoolDType,
                Int8DType,
                Int16DType,
                Int32DType,
                Int64DType,
                Float16DType,
                Float32DType,
                Float64DType,
                Complex64DType,
                Complex128DType,
            ]
        )
    except ImportError:
        pass
    torch.serialization.add_safe_globals(safe_globals)
    return torch.load(path, map_location=map_location, weights_only=True)


def csv_text(header, rows) -> str:
    """Render a small CSV (header + rows) to a string with stdlib quoting."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def resolve_data_path(path) -> str:
    """Expand ``path`` to a normalized absolute path; None raises with a clear message."""
    if path is None:
        raise ValueError(
            "No data path given. Pass --data-path (or set data_path in the YAML config)."
        )
    return os.path.realpath(os.path.expanduser(str(path)))


def runs_from_indices(idx: np.ndarray):
    """Split arbitrary indices into contiguous (start, end) runs.

    Returns ``(runs, sorted_indices, inverse_order)``: concatenating the HDF5
    slices given by ``runs`` yields rows in ``sorted_indices`` order, and
    applying ``inverse_order`` afterwards restores the caller's order.
    """
    idx = np.asarray(idx)
    idx_sorted = np.sort(idx)
    order_in_sorted = np.argsort(idx)
    inv_order = np.argsort(order_in_sorted)

    runs = []
    if idx_sorted.size > 0:
        start = idx_sorted[0]
        prev = start
        for x in idx_sorted[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((start, prev + 1))
                start = x
                prev = x
        runs.append((start, prev + 1))
    return runs, idx_sorted, inv_order


def summarize_score_array(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def normalize_percentile(values: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo = float(np.percentile(values, lower))
    hi = float(np.percentile(values, upper))
    if hi <= lo + 1e-12:
        return np.ones_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def print_score_summary(name: str, values: np.ndarray):
    summary = summarize_score_array(values)
    print(
        f"{name}: mean={summary['mean']:.4f}, std={summary['std']:.4f}, "
        f"p05={summary['p05']:.4f}, p25={summary['p25']:.4f}, "
        f"p50={summary['p50']:.4f}, p75={summary['p75']:.4f}, p95={summary['p95']:.4f}"
    )


def acc(labels, preds):
    """Best label-matched clustering accuracy via the Hungarian algorithm.

    Returns ``(acc_value, correct_indices, incorrect_indices, mapped_preds)``.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    contingency = contingency_matrix(labels, preds, sparse=False)
    if contingency.size == 0 or contingency.sum() == 0:
        return 0.0, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = contingency[row_ind, col_ind].sum()
    acc_value = matched / contingency.sum()

    mapping = -np.ones(contingency.shape[1], dtype=int)
    for row, col in zip(row_ind, col_ind):
        mapping[col] = row
    mapped_preds = mapping[preds]
    correct_mask = mapped_preds == labels
    return acc_value, np.where(correct_mask)[0], np.where(~correct_mask)[0], mapped_preds


def save_confusion_matrix_percent(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_labels,
    out_dir: str,
    filename_prefix: str,
    title: str,
):
    out_dir = checked_path(out_dir)
    maybe_mkdir(out_dir)
    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)
    class_labels = [int(label) for label in class_labels]

    cm = sklearn.metrics.confusion_matrix(true_labels, pred_labels, labels=class_labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm,
        np.maximum(row_sums, 1),
        out=np.zeros_like(cm, dtype=float),
        where=row_sums > 0,
    ) * 100.0

    csv_path = os.path.join(out_dir, f"{filename_prefix}.csv")
    rows = []
    for row_idx, label in enumerate(class_labels):
        rows.append([str(label)] + [f"{value:.4f}" for value in cm_percent[row_idx]])
    text = csv_text(["true/pred"] + [str(label) for label in class_labels], rows)
    Path(csv_path).write_text(text, encoding="utf-8")

    fig, ax = plt.subplots(figsize=CONFUSION_FIGSIZE)
    im = ax.imshow(cm_percent, cmap="Blues", vmin=0.0, vmax=100.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percent", fontsize=CONFUSION_CBAR_FONTSIZE)
    cbar.ax.tick_params(labelsize=CONFUSION_CBAR_FONTSIZE)
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels([str(label) for label in class_labels], fontsize=CONFUSION_TICK_FONTSIZE)
    ax.set_yticklabels([str(label) for label in class_labels], fontsize=CONFUSION_TICK_FONTSIZE)
    ax.set_xlabel("Predicted Label", fontsize=CONFUSION_LABEL_FONTSIZE)
    ax.set_ylabel("True Label", fontsize=CONFUSION_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=CONFUSION_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=CONFUSION_TICK_FONTSIZE)

    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            text_color = "white" if cm_percent[i, j] >= 50.0 else "black"
            ax.text(
                j,
                i,
                f"{cm_percent[i, j]:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=CONFUSION_CELL_FONTSIZE,
            )

    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{filename_prefix}.png")
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def cluster(features, n_clusters=2, method="kmeans"):
    if method == "kmeans":
        cluster_model = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == "gmm":
        cluster_model = sklearn.mixture.GaussianMixture(n_components=n_clusters, random_state=42)
    elif method == "agglomerative":
        cluster_model = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)
    elif method == "dbscan":
        cluster_model = sklearn.cluster.DBSCAN(eps=0.5, min_samples=10)
    elif method == "spectral":
        cluster_model = sklearn.cluster.SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            affinity="nearest_neighbors",
            n_neighbors=15,
            n_init=20,
        )
    elif method == "km_cosine":
        cluster_model = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
        features = sklearn.preprocessing.normalize(features)
    elif method == "agg_cosine":
        cluster_model = sklearn.cluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return cluster_model.fit_predict(features)


def preprocess_cluster_features(features, feature_mean=None, feature_scale=None, eps=1e-12):
    """Standardize per-feature, then L2-normalize each sample."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"features must have shape (n_samples, n_features); got {x.shape}.")

    if feature_mean is None:
        feature_mean = x.mean(axis=0)
    if feature_scale is None:
        feature_scale = x.std(axis=0)

    feature_mean = np.asarray(feature_mean, dtype=float)
    feature_scale = np.asarray(feature_scale, dtype=float)
    safe_scale = np.where(feature_scale > eps, feature_scale, 1.0)
    x = (x - feature_mean) / safe_scale
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)
    return x, feature_mean, safe_scale


def dedim(features, ver="pca"):
    if ver == "pca":
        pca = sklearn.decomposition.PCA(n_components=2)
        return pca.fit_transform(features)
    if ver == "tsne":
        tsne = sklearn.manifold.TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=42,
        )
        return tsne.fit_transform(features)
    raise ValueError(f"Unknown reduction method: {ver}")


def compute_binary_cluster_centers(features: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(cluster_labels, dtype=int)
    uniq = np.sort(np.unique(labels))
    if uniq.size != 2:
        raise ValueError(f"Expected exactly 2 clusters, got {uniq.size}.")
    return np.stack([features[labels == label].mean(axis=0) for label in uniq], axis=0)


def assign_binary_clusters_by_centers(
    features: np.ndarray, centers: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    return labels.astype(np.int64), distances


def compute_cluster_margin(features: np.ndarray, centers: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    _, distances = assign_binary_clusters_by_centers(features, centers)
    d0 = distances[:, 0]
    d1 = distances[:, 1]
    margin = np.abs(d0 - d1) / (d0 + d1 + eps)
    return np.clip(margin, 0.0, 1.0)
