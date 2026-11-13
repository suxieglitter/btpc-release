"""Run a new, seeded prediction realization with the released primary model.

This does not recreate the historical target TTA RNG state. Outputs must be
written outside paper_artifacts so the released records remain unchanged.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

from verify_paper_artifacts import DEFAULT_PACKAGE, ROOT, reference_inference, sha256

sys.path.insert(0, str(ROOT))
from btpc.predict import compute_prediction_metrics, load_stage2_bundle
from btpc.valid import preprocess_clean_waveforms, read_source_slices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--batch", choices=["training", "validation", "consensus"], required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tta-seed", type=int, default=20261113)
    parser.add_argument("--limit", type=int, default=0, help="Positive limit is a diagnostic subset, not the paper batch.")
    args = parser.parse_args()
    package, output = args.package.resolve(), args.output_dir.resolve()
    if output == package or package in output.parents:
        parser.error("Output must be outside the frozen package")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty or absent")
    manifest = json.loads((package / "manifest.json").read_text())
    batch = manifest["batches"][args.batch]
    split_path = package / batch["split"]
    with split_path.open(newline="") as handle:
        identities = list(csv.DictReader(handle))
    if args.limit > 0:
        identities = identities[:args.limit]
    reference, _ = reference_inference(package)
    model_path = package / "checkpoints/seed42/stage2_cluster_classifier.pth"
    model, bundle = load_stage2_bundle(str(model_path))
    cfg = bundle["stage1_config"]
    np.random.seed(args.tta_seed)
    output.mkdir(parents=True, exist_ok=True)
    names = ["source_index", "evid", "sncl", "true_label", "pred_cluster_class",
             "pred_polarity", "mean_confidence", "mean_center_margin", "is_reject", "final_label"]
    correct = accepted = correct_accepted = 0
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for start in range(0, len(identities), 8192):
            chunk = identities[start:start + 8192]
            indices = np.asarray([int(r["source_index"]) for r in chunk])
            raw, labels, _, evids, sncls = read_source_slices(
                str(args.data_path), indices, waveform_key="X", label_key="Y", snr_key="snr",
            )
            for i, identity in enumerate(chunk):
                if int(labels[i]) != int(identity["label"]) or str(evids[i]) != identity["evid"] or str(sncls[i]) != identity["sncl"]:
                    raise ValueError(f"Input HDF5 differs from released identity at row {start + i}")
            waves = preprocess_clean_waveforms(raw, resize=int(cfg["resize"]), shift=int(cfg["shift"]), norm_mod=cfg["norm_mod"])
            measured = compute_prediction_metrics(
                model=model, clean_waveforms=waves,
                cluster_centers=np.asarray(bundle["cluster_centers"], dtype=np.float32),
                feature_mean=np.asarray(bundle["feature_mean"], dtype=np.float32),
                feature_scale=np.asarray(bundle["feature_scale"], dtype=np.float32),
                n_tta_views=30, tta_max_shift=1, batch_size=int(cfg["batch_size"]),
                tta_scale_jitter=0.02, tta_noise_std=0.015,
            )
            for i, identity in enumerate(chunk):
                cluster = int(measured["modal_class"][i])
                pred = int(reference["cluster_mapping"][str(cluster)])
                mc, mcm = float(measured["mean_confidence"][i]), float(measured["mean_center_margin"][i])
                reject = mc < 0.8 and mcm < 0.05
                truth = int(labels[i])
                correct += truth == pred
                accepted += not reject
                correct_accepted += truth == pred and not reject
                writer.writerow(dict(zip(names, [identity["source_index"], identity["evid"], identity["sncl"], truth, cluster, pred, mc, mcm, int(reject), 2 if reject else pred])))
    summary = {
        "scope": "new seeded TTA realization, not the frozen paper realization",
        "batch": args.batch, "record_count": len(identities), "limit": args.limit,
        "tta_seed": args.tta_seed, "tta_settings": reference["tta_settings"], "chunk_size": 8192,
        "current_reference": reference, "checkpoint_sha256": sha256(model_path),
        "split_sha256": sha256(split_path), "data_path": str(args.data_path.resolve()),
        "accuracy_before_reject": correct / len(identities), "accept_rate": accepted / len(identities),
        "accuracy_on_accepted_only": correct_accepted / accepted if accepted else None,
        "environment": {"torch": torch.__version__, "numpy": np.__version__},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
