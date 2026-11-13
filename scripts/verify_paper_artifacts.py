"""Verify the frozen SCEDC primary results without rerunning target inference.

The default check uses only the Python standard library. --reference-inference
also runs the released classifier on the 200 reference records with a fixed
TTA seed and requires the repository's scientific Python dependencies.
"""

import argparse
import csv
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "paper_artifacts/scedc_primary_20261113"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_predictions(predictions, split):
    """Recount predictions and validate identity, mapping and rejection per row."""
    n = correct = accepted = correct_accepted = 0
    confusion = [[0, 0, 0], [0, 0, 0]]
    with Path(predictions).open(newline="") as ph, Path(split).open(newline="") as sh:
        for row, identity in zip_longest(csv.DictReader(ph), csv.DictReader(sh)):
            if row is None or identity is None:
                raise ValueError("Prediction and split row counts differ")
            for key in ("source_index", "evid", "sncl"):
                if row[key] != identity[key]:
                    raise ValueError(f"Identity mismatch at row {n}: {key}")
            truth = int(row["true_label"])
            if truth != int(identity["label"]):
                raise ValueError(f"Manual label mismatch at row {n}")
            pred = int(row["pred_polarity"])
            if pred != 1 - int(row["pred_cluster_class"]):
                raise ValueError(f"Cluster-to-polarity mapping mismatch at row {n}")
            confidence, margin = float(row["mean_confidence"]), float(row["mean_center_margin"])
            if not all(map(math.isfinite, (confidence, margin))):
                raise ValueError(f"Non-finite score at row {n}")
            reject = confidence < 0.8 and margin < 0.05
            if int(row["is_reject"]) != int(reject):
                raise ValueError(f"AND rejection mismatch at row {n}")
            final = int(row["final_label"])
            if final != (2 if reject else pred):
                raise ValueError(f"Final label mismatch at row {n}")
            n += 1
            correct += truth == pred
            accepted += not reject
            correct_accepted += (truth == pred) and not reject
            confusion[truth][final] += 1
    return {
        "record_count": n,
        "correct_before_rejection": correct,
        "accepted_count": accepted,
        "correct_accepted_count": correct_accepted,
        "rejected_count": n - accepted,
        "accuracy_before_reject": correct / n,
        "accept_rate": accepted / n,
        "accuracy_on_accepted_only": correct_accepted / accepted,
        "accuracy_after_reject_reject_as_wrong": correct_accepted / n,
        "confusion_true_up_down_by_final_up_down_reject": confusion,
    }


def reference_inference(package):
    """Run the current fixed reference draw; do not alter any target predictions."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from btpc.anchor_utils import apply_cluster_mapping, fit_anchor_cluster_mapping
    from btpc.anchor_utils import load_anchor_bank, prepare_anchor_waveforms_for_ridge
    from btpc.predict import compute_prediction_metrics, load_stage2_bundle

    torch.set_num_threads(min(4, torch.get_num_threads()))
    reference_path = package / "data_splits/reference_200.csv"
    with reference_path.open(newline="") as handle:
        refs = list(csv.DictReader(handle))
    bank = load_anchor_bank(str(package / "reference/anchor_bank_native.npz"))
    def read_split(name):
        with (package / "data_splits" / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    train = read_split("native_train_10k.csv")
    valid = read_split("native_valid_100k.csv")
    consensus = read_split("consensus_100k.csv")
    train_rows = {int(r["source_index"]) for r in train}
    train_events = {int(r["evid"]) for r in train}
    valid_rows = {int(r["source_index"]) for r in valid}
    excluded_rows = train_rows | valid_rows
    consensus_keys = {(r["evid"], r["sncl"]) for r in consensus}
    with (package / "reference/bank_metadata.csv").open(newline="") as handle:
        metadata = list(csv.DictReader(handle))
    if not np.array_equal(bank["source_indices"], [int(r["source_index"]) for r in metadata]):
        raise ValueError("Reference bank metadata order differs")
    eligible = np.asarray([
        int(r["source_index"]) not in excluded_rows
        and int(r["evid"]) not in train_events
        and (r["evid"], r["sncl"]) not in consensus_keys for r in metadata
    ])
    rng = np.random.default_rng(20261004)
    chosen = np.concatenate([
        rng.choice(np.where((bank["labels"] == label) & eligible)[0], size=100, replace=False)
        for label in (0, 1)
    ])
    chosen = chosen[rng.permutation(chosen.size)]
    if sorted(int(x) for x in bank["source_indices"][chosen]) != sorted(int(r["source_index"]) for r in refs):
        raise ValueError("Reference exclusion and sampling replay differs from published rows")
    # Inference must use the historical randomized draw order, not the sorted CSV.
    by_source = {int(r["source_index"]): r for r in refs}
    refs = [by_source[int(source)] for source in bank["source_indices"][chosen]]
    labels = np.asarray(bank["labels"])[chosen].astype(np.int64)
    if not np.array_equal(labels, [int(r["label"]) for r in refs]):
        raise ValueError("Reference bank and published reference labels differ")
    model, bundle = load_stage2_bundle(str(package / "checkpoints/seed42/stage2_cluster_classifier.pth"))
    cfg = bundle["stage1_config"]
    if cfg != json.loads((package / "checkpoints/seed42/stage1_config.json").read_text()):
        raise ValueError("Checkpoint and published Stage-1 configuration differ")
    waves = prepare_anchor_waveforms_for_ridge(
        np.asarray(bank["raw_waveforms"])[chosen], resize=int(cfg["resize"]),
        shift=int(cfg["shift"]), p_arrival_index=int(bank["p_arrival_index"]),
    )
    np.random.seed(20261004)
    measured = compute_prediction_metrics(
        model=model, clean_waveforms=waves,
        cluster_centers=np.asarray(bundle["cluster_centers"], dtype=np.float32),
        feature_mean=np.asarray(bundle["feature_mean"], dtype=np.float32),
        feature_scale=np.asarray(bundle["feature_scale"], dtype=np.float32),
        n_tta_views=30, tta_max_shift=1, batch_size=int(cfg["batch_size"]),
        tta_scale_jitter=0.02, tta_noise_std=0.015,
    )
    modal = measured["modal_class"]
    mapping, stats = fit_anchor_cluster_mapping(
        cluster_labels=modal, is_anchor=np.ones_like(modal), known_labels=labels,
    )
    mapping = {int(k): int(v) for k, v in mapping.items()}
    mapped = apply_cluster_mapping(modal, mapping)
    correct = int(np.sum(mapped == labels))
    if mapping != {0: 1, 1: 0} or correct != 192:
        raise ValueError(f"Reference result differs: mapping={mapping}, correct={correct}/200")
    # Stage 2 freezes this exact Stage-1 encoder; compare tensors, not filenames.
    stage1 = torch.load(package / "checkpoints/seed42/final_pwave_model.pth", map_location="cpu", weights_only=False)
    expected = {k.removeprefix("encoder."): v for k, v in stage1["model_state_dict"].items() if k.startswith("encoder.")}
    actual = bundle["encoder_state_dict"]
    if set(expected) != set(actual) or any(not torch.equal(expected[k].cpu(), actual[k].cpu()) for k in expected):
        raise ValueError("Stage-1 and Stage-2 encoder tensors differ")
    rows = [
        {"source_index": int(ref["source_index"]), "true_label": int(label),
         "pred_cluster_class": int(cluster), "pred_polarity": int(pred),
         "correct": int(label == pred)}
        for ref, label, cluster, pred in zip(refs, labels, modal, mapped)
    ]
    return {
        "reference_count": len(refs), "correct_count": correct,
        "mapping_accuracy": correct / len(refs),
        "cluster_mapping": {str(k): v for k, v in mapping.items()},
        "reference_selection_seed": 20261004, "tta_seed": 20261004,
        "reference_exclusion_and_draw_replayed": True,
        "reference_inference_order": "seeded draw order before source-index sorting",
        "eligible_bank_count": int(eligible.sum()),
        "tta_settings": {"n_tta_views": 30, "tta_max_shift": 1,
                         "tta_scale_jitter": 0.02, "tta_noise_std": 0.015},
        "stage1_encoder_matches_stage2": True,
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "device": str(next(model.parameters()).device)},
    }, rows


def verify(package, run_reference=False):
    manifest = json.loads((package / "manifest.json").read_text())
    actual_files = {str(p.relative_to(package)) for p in package.rglob("*") if p.is_file() and p != package / "manifest.json"}
    if actual_files != {entry["path"] for entry in manifest["files"]}:
        raise ValueError("Package file list differs from manifest")
    for entry in manifest["files"]:
        path = package / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Integrity mismatch: {entry['path']}")
    for entry in manifest["code_files"]:
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Code version differs from release: {entry['path']}")
    results = {}
    for name, batch in manifest["batches"].items():
        result = summarize_predictions(package / batch["predictions"], package / batch["split"])
        summary = json.loads((package / batch["summary"]).read_text())
        if result != summary["metrics"]:
            raise ValueError(f"Recount differs from published summary: {name}")
        if summary["current_reference"]["mapping_accuracy"] != 0.96:
            raise ValueError(f"Incorrect current reference in {name}")
        results[name] = result
    with (package / "reference/reference_predictions.csv").open(newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    reference_summary = json.loads((package / "reference/reference_summary.json").read_text())
    correct = sum(int(r["true_label"]) == int(r["pred_polarity"]) for r in reference_rows)
    if len(reference_rows) != 200 or correct != 192 or reference_summary["correct_count"] != correct:
        raise ValueError("Reference recount differs from 192/200")
    if run_reference:
        replay, rows = reference_inference(package)
        if any({k: int(v) for k, v in old.items()} != new for old, new in zip(reference_rows, rows)):
            raise ValueError("Reference replay row predictions differ")
        results["reference_replay"] = replay
    results["verified_files"] = len(manifest["files"])
    results["verified_code_files"] = len(manifest["code_files"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--reference-inference", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.package.resolve(), args.reference_inference), indent=2))


if __name__ == "__main__":
    main()
