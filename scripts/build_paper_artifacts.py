"""Package existing SCEDC primary artifacts; no training or target inference.

This maintainer utility needs the local native_v2 runs. Readers use
verify_paper_artifacts.py against the committed package instead.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from verify_paper_artifacts import DEFAULT_PACKAGE, ROOT, reference_inference
from verify_paper_artifacts import sha256, summarize_predictions


BATCHES = {
    "training": ("valid_train10k", "native_train_10k.csv"),
    "validation": ("valid_native100k", "native_valid_100k.csv"),
    "consensus": ("valid_consensus100k", "consensus_100k.csv"),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    package = DEFAULT_PACKAGE
    run = ROOT / "runs/native_v2"
    copied = {}

    def copy(source, relative):
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied[relative] = {"source": str(source.relative_to(ROOT)), "sha256": sha256(source)}

    for directory, names in {
        "stage1": ["final_pwave_model.pth", "stage1_config.json", "stage1_summary.json"],
        "stage2": ["stage2_cluster_classifier.pth", "stage2_summary.json", "pseudo_labels.csv"],
    }.items():
        for name in names:
            copy(run / directory / name, f"checkpoints/seed42/{name}")
    for name in ("native_train_10k.csv", "native_valid_100k.csv", "consensus_100k.csv",
                 "reference_200.csv", "split_summary.csv", "manifest.json"):
        copy(ROOT / "data_splits" / name, f"data_splits/{name}")
    for name in ("anchor_bank_native.npz", "anchor_bank_native_manifest.csv"):
        copy(run / name, f"reference/{name}")
    # Metadata enables replay of the reference exclusions without the full HDF5.
    import h5py
    import numpy as np
    split_manifest = json.loads((ROOT / "data_splits/manifest.json").read_text())
    with np.load(run / "anchor_bank_native.npz", allow_pickle=False) as bank:
        source_indices = bank["source_indices"]
        bank_labels = bank["labels"]
    with h5py.File(split_manifest["native_path"], "r") as source:
        evids = np.asarray(source["evids"][:])[source_indices]
        sncls = np.asarray(source["sncls"][:])[source_indices]
    with (package / "reference/bank_metadata.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_index", "evid", "sncl", "label"])
        for source_index, evid, sncl, label in zip(source_indices, evids, sncls, bank_labels):
            writer.writerow([int(source_index), int(evid), sncl.decode() if isinstance(sncl, bytes) else str(sncl), int(label)])
    copy(run / "valid_dedup_swap_20261014/final_selection.json", "provenance/validation_final_selection.json")
    with (run / "valid_dedup_swap_20261014/native_valid_100k.before_swap.csv").open(newline="") as handle:
        before = list(csv.DictReader(handle))
    with (ROOT / "data_splits/native_valid_100k.csv").open(newline="") as handle:
        after = list(csv.DictReader(handle))
    replacements = [{"position": i, "old": old, "new": new} for i, (old, new) in enumerate(zip(before, after)) if old["source_index"] != new["source_index"]]
    if len(before) != len(after) or len(replacements) != 8:
        raise ValueError("Expected exactly eight realized validation replacements")
    write_json(package / "provenance/validation_realized_replacements.json", {
        "source_before_sha256": sha256(run / "valid_dedup_swap_20261014/native_valid_100k.before_swap.csv"),
        "source_after_sha256": sha256(ROOT / "data_splits/native_valid_100k.csv"),
        "replacements": replacements,
        "note": "Realized records are compared directly; an earlier swap_record.json contains superseded choices and is not this release.",
    })
    reference, rows = reference_inference(package)
    write_json(package / "reference/reference_summary.json", reference)
    with (package / "reference/reference_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    batches = {}
    for name, (source_dir, split_file) in BATCHES.items():
        prediction = f"predictions/{name}/valid_predictions.csv"
        summary_path = f"predictions/{name}/valid_summary.json"
        split = f"data_splits/{split_file}"
        original_path = f"provenance/{name}_inference_summary_original.json"
        copy(run / source_dir / "valid_predictions.csv", prediction)
        copy(run / source_dir / "valid_summary.json", original_path)
        original = json.loads((package / original_path).read_text())
        if original["stage2_checkpoint"] != "runs/native_v2/stage2/stage2_cluster_classifier.pth":
            raise ValueError(f"Unexpected checkpoint provenance for {name}")
        metrics = summarize_predictions(package / prediction, package / split)
        for key in ("accuracy_before_reject", "accept_rate", "accuracy_on_accepted_only"):
            if abs(metrics[key] - original["metrics"][key]) > 1e-12:
                raise ValueError(f"Original summary disagrees with recount: {name} {key}")
        write_json(package / summary_path, {
            "schema_version": 1, "batch": name,
            "stage2_checkpoint": "checkpoints/seed42/stage2_cluster_classifier.pth",
            "stage2_checkpoint_sha256": sha256(package / "checkpoints/seed42/stage2_cluster_classifier.pth"),
            "predictions": prediction, "predictions_sha256": sha256(package / prediction),
            "split": split, "split_sha256": sha256(package / split),
            "metrics": metrics, "tta_settings": original["tta_settings"],
            "reject_rule": "(mean_confidence < 0.8) AND (mean_center_margin < 0.05)",
            "cluster_mapping": original["cluster_mapping"],
            "prediction_time_reference": {
                "mapping_accuracy": original["mapping_accuracy"],
                "mapping_info": original["mapping_info"],
                "cluster_mapping": original["cluster_mapping"],
                "original_summary": original_path,
            },
            "current_reference": {
                "split": "data_splits/reference_200.csv", "summary": "reference/reference_summary.json",
                "mapping_accuracy": reference["mapping_accuracy"],
                "cluster_mapping": reference["cluster_mapping"],
                "selection_seed": 20261004, "tta_seed": 20261004,
            },
            "provenance_limit": "Existing target CSVs are preserved and recounted, not regenerated. Historical target TTA RNG states were not saved. The new reference draw preserves the global mapping; it does not recreate the target TTA draws.",
        })
        batches[name] = {"predictions": prediction, "summary": summary_path, "split": split}
    write_json(package / "manifest.json", {
        "schema_version": 1, "package_id": "scedc_primary_20261113",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Primary native SCEDC seed42 experiment: training, validation, consensus and current reference mapping. No additional experiments are claimed by this package.",
        "batches": batches, "source_copies": copied,
        "code_files": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in
                       sorted(list((ROOT / "btpc").glob("*.py")) + [
                           ROOT / "scripts/verify_paper_artifacts.py",
                           ROOT / "scripts/replay_paper_predictions.py",
                           ROOT / "scripts/build_paper_artifacts.py",
                           ROOT / "scripts/make_event_splits.py",
                           ROOT / "scripts/redraw_references.py",
                           ROOT / "requirements.txt", ROOT / "pyproject.toml",
                       ])],
        "files": [{"path": str(p.relative_to(package)), "bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in sorted(package.rglob("*")) if p.is_file() and p != package / "manifest.json"],
    })
    print(json.dumps({"package": str(package), "reference": reference, "batches": batches}, indent=2))


if __name__ == "__main__":
    main()
