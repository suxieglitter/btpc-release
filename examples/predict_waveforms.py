"""Apply the released SCEDC model to unlabeled, aligned waveforms.

Input: an NPZ with waveforms shaped (N, 600), sampled at 100 Hz, P at index 300.
Run from the repository root after installing the package with pip install -e .
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_paper_artifacts import DEFAULT_PACKAGE, reference_inference, sha256
from btpc.predict import compute_prediction_metrics, load_stage2_bundle
from btpc.valid import preprocess_clean_waveforms


def load_waveforms(path):
    with np.load(path, allow_pickle=False) as data:
        if "waveforms" not in data:
            raise ValueError("Input NPZ must contain a 'waveforms' array")
        waves = data["waveforms"]
    if waves.ndim != 2 or waves.shape[0] == 0 or waves.shape[1] != 600:
        raise ValueError("waveforms must have shape (N, 600), with N > 0; supply full traces, not cropped windows")
    if waves.dtype.kind not in "fiu" or not np.isfinite(waves).all():
        raise ValueError("waveforms must contain finite real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        waves = waves.astype(np.float32)
    if not np.isfinite(waves).all():
        raise ValueError("waveforms must be representable as finite float32 values")
    return waves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NPZ containing your 'waveforms' array")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tta-seed", type=int, default=20261113)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == DEFAULT_PACKAGE or DEFAULT_PACKAGE in output.parents:
        parser.error("Output must be outside the frozen paper package")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be empty or absent")
    try:
        waves = load_waveforms(args.input)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    reference, _ = reference_inference(DEFAULT_PACKAGE)
    checkpoint = DEFAULT_PACKAGE / "checkpoints/seed42/stage2_cluster_classifier.pth"
    model, bundle = load_stage2_bundle(str(checkpoint))
    cfg = bundle["stage1_config"]
    tta = dict(n_tta_views=30, tta_max_shift=1, tta_scale_jitter=0.02, tta_noise_std=0.015)
    # The reference check sets its own seed; seed target TTA only after it finishes.
    np.random.seed(args.tta_seed)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["input_row", "pred_polarity", "mean_confidence", "mean_center_margin",
              "is_reject", "final_label", "final_label_name"]
    names = {0: "Up", 1: "Down", 2: "Uncertain"}
    accepted = 0
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for start in range(0, len(waves), 8192):
            clean = preprocess_clean_waveforms(
                waves[start:start + 8192], resize=int(cfg["resize"]),
                shift=int(cfg["shift"]), norm_mod=cfg["norm_mod"],
                sampling_rate=100.0, p_arrival_index=300,
            )
            measured = compute_prediction_metrics(
                model=model, clean_waveforms=clean,
                cluster_centers=np.asarray(bundle["cluster_centers"], dtype=np.float32),
                feature_mean=np.asarray(bundle["feature_mean"], dtype=np.float32),
                feature_scale=np.asarray(bundle["feature_scale"], dtype=np.float32),
                batch_size=int(cfg["batch_size"]), **tta,
            )
            for i, cluster in enumerate(measured["modal_class"]):
                pred = int(reference["cluster_mapping"][str(int(cluster))])
                mc = float(measured["mean_confidence"][i])
                mcm = float(measured["mean_center_margin"][i])
                reject = mc < 0.8 and mcm < 0.05
                final = 2 if reject else pred
                accepted += not reject
                writer.writerow(dict(zip(fields, [start + i, pred, mc, mcm, int(reject), final, names[final]])))
    summary = {
        "input": str(args.input.resolve()), "record_count": len(waves),
        "accepted_count": accepted, "coverage": accepted / len(waves),
        "checkpoint_sha256": sha256(checkpoint),
        "cluster_mapping": reference["cluster_mapping"],
        "preprocessing": {"sampling_rate_hz": 100, "p_arrival_index": 300,
                          "detrend_full_trace": True, "resize": int(cfg["resize"]),
                          "shift": int(cfg["shift"]), "norm_mod": cfg["norm_mod"]},
        "tta_seed": args.tta_seed, "tta_settings": tta, "chunk_size": 8192,
        "reject_rule": "(mean_confidence < 0.8) AND (mean_center_margin < 0.05)",
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "device": str(next(model.parameters()).device)},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
