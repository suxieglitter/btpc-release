"""Re-draw the 200 SCEDC reference samples with full batch independence.

Reviewer items R1.4 / R2.2: the original reference draw (sample seed 20260404)
shared one record with the training batch. This script re-draws the
100-up / 100-down reference subset from the committed anchor bank
(runs/native_v2/anchor_bank_native.npz, unchanged) with a new seed, excluding

- training-batch rows and training-batch events (record- and event-level),
- validation-batch rows,
- consensus-batch records (matched by evid|sncl),

then recomputes the cluster-to-polarity mapping and its accuracy with the
native_v2 Stage-2 model via the same code path as ``btpc-valid``
(fit_anchor_cluster_mapping on TTA-view modal classes), with the TTA noise
seeded explicitly so the number is reproducible (the original 0.98 came from
an unseeded global RNG state).

The mapping orientation is asserted equal to the published one
(cluster 0 -> down, cluster 1 -> up); because target-batch predictions depend
on the reference set only through this orientation, every downstream metric
(training/validation/consensus accuracies, flip experiment, Table 1) is
unchanged by construction. Only the mapping-accuracy number and the
reference-set descriptive statistics can change.

Outputs (with --apply):
- data_splits/reference_200.csv        replaced by the new 200 rows
- data_splits/split_summary.csv        reference_200 row updated
- data_splits/manifest.json            reference section updated with the
                                       redraw provenance and exclusions

All other split files are left untouched. Without --apply the script prints
the draw and the mapping check only.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from btpc.anchor_utils import (  # noqa: E402
    apply_cluster_mapping,
    fit_anchor_cluster_mapping,
    load_anchor_bank,
    prepare_anchor_waveforms_for_ridge,
)
from btpc.predict import compute_prediction_metrics, load_stage2_bundle  # noqa: E402

# Published evaluation settings of the native_v2 SCEDC experiment
# (runs/native_v2/valid_train10k/valid_summary.json).
TTA_SETTINGS = {
    "n_tta_views": 30,
    "tta_max_shift": 1,
    "tta_scale_jitter": 0.02,
    "tta_noise_std": 0.015,
}
PUBLISHED_MAPPING = {0: 1, 1: 0}


def decode_sncls(values):
    decoded = []
    for value in np.asarray(values):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(value))
    return np.asarray(decoded, dtype=object)


def draw_excluded_subset(bank, mask, n_per_label, sample_seed):
    """Mirror btpc.anchor_utils.sample_anchor_subset, restricted to ``mask``."""
    labels = np.asarray(bank["labels"], dtype=np.int64)
    rng = np.random.default_rng(int(sample_seed))
    selected = []
    for label in (0, 1):
        eligible = np.where((labels == label) & mask)[0]
        if eligible.size < int(n_per_label):
            raise ValueError(
                f"Excluded bank has only {eligible.size} rows for label={label}; "
                f"need {n_per_label}."
            )
        selected.append(rng.choice(eligible, size=int(n_per_label), replace=False))
    chosen = np.concatenate(selected, axis=0)
    chosen = chosen[rng.permutation(chosen.size)]
    subset = {
        "raw_waveforms": np.asarray(bank["raw_waveforms"][chosen], dtype=np.float32),
        "labels": np.asarray(bank["labels"][chosen], dtype=np.int64),
        "snrs": np.asarray(bank["snrs"][chosen], dtype=np.float32),
        "source_indices": np.asarray(bank["source_indices"][chosen], dtype=np.int64),
        "p_arrival_index": int(np.asarray(bank["p_arrival_index"]).item()),
        "seed": int(sample_seed),
    }
    return subset, chosen


def evaluate_mapping(model, bundle, subset, tta_seed):
    """Cluster-to-polarity mapping on the subset, via the btpc-valid code path."""
    cfg = bundle["stage1_config"]
    waves = prepare_anchor_waveforms_for_ridge(
        subset["raw_waveforms"],
        resize=int(cfg["resize"]),
        shift=int(cfg["shift"]),
        p_arrival_index=int(subset["p_arrival_index"]),
    )
    np.random.seed(int(tta_seed))  # TTA augmentation reads the global RNG
    metrics = compute_prediction_metrics(
        model=model,
        clean_waveforms=waves,
        cluster_centers=np.asarray(bundle["cluster_centers"], dtype=np.float32),
        feature_mean=np.asarray(bundle["feature_mean"], dtype=np.float32),
        feature_scale=np.asarray(bundle["feature_scale"], dtype=np.float32),
        n_tta_views=TTA_SETTINGS["n_tta_views"],
        tta_max_shift=TTA_SETTINGS["tta_max_shift"],
        batch_size=int(cfg["batch_size"]),
        tta_scale_jitter=TTA_SETTINGS["tta_scale_jitter"],
        tta_noise_std=TTA_SETTINGS["tta_noise_std"],
    )
    modal = np.asarray(metrics["modal_class"], dtype=np.int64)
    mapping, cluster_stats = fit_anchor_cluster_mapping(
        cluster_labels=modal,
        is_anchor=np.ones_like(modal),
        known_labels=np.asarray(subset["labels"], dtype=np.int64),
    )
    mapping_int = {int(k): int(v) for k, v in mapping.items()}
    mapped = apply_cluster_mapping(modal, mapping)
    accuracy = float(
        np.mean(mapped == np.asarray(subset["labels"], dtype=np.int64))
    )
    return mapping_int, cluster_stats, accuracy


def reference_summary_row(native_path, source_indices):
    """Descriptive statistics in the split_summary.csv convention."""
    with h5py.File(str(native_path), "r") as handle:
        labels = np.asarray(handle["Y"][:])[source_indices]
        snrs = np.asarray(handle["snr"][:], dtype=np.float64)[source_indices]
        evids = np.asarray(handle["evids"][:])[source_indices]
        sncls = decode_sncls(np.asarray(handle["sncls"][:])[source_indices])
        mags = np.asarray(handle["mag"][:], dtype=np.float64)[source_indices]
        dists = np.asarray(handle["dist"][:], dtype=np.float64)[source_indices]
    stations = [str(s).split(".")[1] if "." in str(s) else str(s) for s in sncls]
    return {
        "split": "reference_200",
        "n": int(labels.size),
        "n_up": int(np.sum(labels == 0)),
        "n_down": int(np.sum(labels == 1)),
        "up_fraction": float(np.mean(labels == 0)),
        "n_events": int(np.unique(evids).size),
        "n_stations": int(np.unique(stations).size),
        "snr_mean": float(np.mean(snrs)),
        "snr_p10": float(np.percentile(snrs, 10)),
        "snr_p25": float(np.percentile(snrs, 25)),
        "snr_p50": float(np.percentile(snrs, 50)),
        "snr_p75": float(np.percentile(snrs, 75)),
        "snr_p90": float(np.percentile(snrs, 90)),
        "mag_p50": float(np.percentile(mags, 50)),
        "mag_mean": float(np.mean(mags)),
        "dist_p50": float(np.percentile(dists, 50)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bank", default=str(REPO_ROOT / "runs/native_v2/anchor_bank_native.npz")
    )
    parser.add_argument(
        "--stage2-checkpoint",
        default=str(REPO_ROOT / "runs/native_v2/stage2/stage2_cluster_classifier.pth"),
    )
    parser.add_argument("--splits-dir", default=str(REPO_ROOT / "data_splits"))
    parser.add_argument(
        "--native-path",
        default=None,
        help="Native HDF5; defaults to the native_path recorded in manifest.json.",
    )
    parser.add_argument("--n-per-label", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=20261004)
    parser.add_argument("--tta-seed", type=int, default=20261004)
    parser.add_argument(
        "--verify-published-draw",
        action="store_true",
        help="Also replay the original seed-20260404 subset and report its mapping "
        "(no exclusions); orientation must match the published one.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write reference_200.csv, split_summary.csv and manifest.json. "
        "Default: dry run.",
    )
    args = parser.parse_args(argv)

    splits = Path(args.splits_dir)
    manifest = json.loads((splits / "manifest.json").read_text())
    native_path = args.native_path or manifest["native_path"]

    train = pd.read_csv(splits / "native_train_10k.csv")
    valid = pd.read_csv(splits / "native_valid_100k.csv")
    consensus = pd.read_csv(splits / "consensus_100k.csv")

    train_rows = set(train["source_index"].astype(np.int64))
    train_events = set(train["evid"].astype(np.int64))
    valid_rows = set(valid["source_index"].astype(np.int64))
    consensus_keys = {
        f"{int(e)}|{s}" for e, s in zip(consensus["evid"], consensus["sncl"])
    }

    bank = load_anchor_bank(str(args.bank))
    bank_rows = np.asarray(bank["source_indices"], dtype=np.int64)
    with h5py.File(str(native_path), "r") as handle:
        bank_evids = np.asarray(handle["evids"][:])[bank_rows]
        bank_sncls = decode_sncls(np.asarray(handle["sncls"][:])[bank_rows])
    bank_keys = np.asarray(
        [f"{int(e)}|{s}" for e, s in zip(bank_evids, bank_sncls)], dtype=object
    )

    hit_train_rows = np.isin(bank_rows, list(train_rows))
    hit_train_events = np.isin(bank_evids, list(train_events))
    hit_valid_rows = np.isin(bank_rows, list(valid_rows))
    hit_consensus = np.isin(bank_keys, list(consensus_keys))
    mask = ~(hit_train_rows | hit_train_events | hit_valid_rows | hit_consensus)

    print(
        f"Bank pool: {bank_rows.size} rows; excluded train-rows={int(hit_train_rows.sum())}, "
        f"train-events(extra)={int((hit_train_events & ~hit_train_rows).sum())}, "
        f"valid-rows={int(hit_valid_rows.sum())}, consensus-records={int(hit_consensus.sum())}; "
        f"eligible={int(mask.sum())}"
    )

    subset, _ = draw_excluded_subset(bank, mask, args.n_per_label, args.sample_seed)

    model, bundle = load_stage2_bundle(str(args.stage2_checkpoint))
    mapping, cluster_stats, accuracy = evaluate_mapping(
        model, bundle, subset, args.tta_seed
    )
    print(f"New reference draw (seed {args.sample_seed}): mapping={mapping}, "
          f"mapping_accuracy={accuracy:.4f}")
    if mapping != PUBLISHED_MAPPING:
        raise SystemExit(
            f"ABORT: mapping orientation changed ({mapping} != published "
            f"{PUBLISHED_MAPPING}); downstream numbers would NOT be invariant."
        )
    print("Mapping orientation unchanged: all downstream metrics are invariant.")

    if args.verify_published_draw:
        old_subset, _ = draw_excluded_subset(
            bank, np.ones(bank_rows.size, dtype=bool), args.n_per_label, 20260404
        )
        old_mapping, _, old_accuracy = evaluate_mapping(
            model, bundle, old_subset, args.tta_seed
        )
        print(f"Published draw replay (seed 20260404): mapping={old_mapping}, "
              f"mapping_accuracy={old_accuracy:.4f} (published 0.98; TTA now seeded, "
              "small variation expected)")

    row = reference_summary_row(native_path, subset["source_indices"])
    print("New reference summary:", json.dumps(row, indent=1))

    # Independence assertions (the point of the redraw)
    new_rows = set(subset["source_indices"].tolist())
    assert not (new_rows & train_rows), "row overlap with training batch"
    assert not (new_rows & valid_rows), "row overlap with validation batch"
    with h5py.File(str(native_path), "r") as handle:
        new_evids = set(
            np.asarray(handle["evids"][:])[np.sort(subset["source_indices"])].tolist()
        )
    assert not (new_evids & train_events), "event overlap with training batch"
    print("Independence checks passed: no row/event overlap with the training batch, "
          "no row overlap with the validation or consensus batches.")

    if not args.apply:
        print("Dry run; re-run with --apply to write the split files.")
        return

    # reference_200.csv in the committed column convention
    columns = [
        "source_index", "evid", "sncl", "label", "label_name", "snr", "mag", "dist",
    ]
    label_names = {0: "up", 1: "down"}
    order = np.argsort(subset["source_indices"])
    src_sorted = np.asarray(subset["source_indices"])[order]
    with h5py.File(str(native_path), "r") as handle:
        meta = {
            "evid": np.asarray(handle["evids"][:])[src_sorted],
            "sncl": decode_sncls(np.asarray(handle["sncls"][:])[src_sorted]),
            "label": np.asarray(handle["Y"][:])[src_sorted],
            "snr": np.asarray(handle["snr"][:], dtype=np.float64)[src_sorted],
            "mag": np.asarray(handle["mag"][:], dtype=np.float64)[src_sorted],
            "dist": np.asarray(handle["dist"][:], dtype=np.float64)[src_sorted],
        }
    out = pd.DataFrame(
        {
            "source_index": src_sorted.astype(np.int64),
            "evid": meta["evid"].astype(np.int64),
            "sncl": meta["sncl"].astype(str),
            "label": meta["label"].astype(np.int64),
            "label_name": [label_names[int(v)] for v in meta["label"]],
            "snr": meta["snr"],
            "mag": meta["mag"],
            "dist": meta["dist"],
        }
    )
    out = out[columns]
    out.to_csv(splits / "reference_200.csv", index=False)

    # split_summary.csv: replace the reference_200 row, keep column order
    summary_path = splits / "split_summary.csv"
    summary = pd.read_csv(summary_path)
    summary = summary[summary["split"] != "reference_200"]
    new_row = pd.DataFrame([row])[summary.columns.tolist()]
    summary = pd.concat([summary, new_row], ignore_index=True)
    summary.to_csv(summary_path, index=False)

    # manifest.json: reference section + summary row, with redraw provenance
    manifest["reference"] = {
        "size": int(row["n"]),
        "per_label": int(args.n_per_label),
        "bank_path": manifest["reference"].get("bank_path", str(args.bank)),
        "bank_snr_min": manifest["reference"].get("bank_snr_min", 50.0),
        "bank_pool_per_label": manifest["reference"].get("bank_pool_per_label", 5000),
        "bank_build_seed": manifest["reference"].get("bank_build_seed", 42),
        "subset_sample_seed": int(args.sample_seed),
        "overlap_with_train_rows": 0,
        "redraw": {
            "date": "2026-10-04",
            "supersedes_sample_seed": 20260404,
            "reason": "reviewer items R1.4/R2.2: eliminate the 1-row overlap with "
            "the training batch and make the reference set independent at both "
            "record and event level",
            "exclusions": [
                "training-batch rows",
                "training-batch events",
                "validation-batch rows",
                "consensus-batch records (evid|sncl)",
            ],
            "excluded_pool_rows": {
                "train_rows": int(hit_train_rows.sum()),
                "train_events_extra": int((hit_train_events & ~hit_train_rows).sum()),
                "valid_rows": int(hit_valid_rows.sum()),
                "consensus_records": int(hit_consensus.sum()),
            },
            "mapping": {str(k): v for k, v in mapping.items()},
            "mapping_accuracy": accuracy,
            "tta_settings": TTA_SETTINGS,
            "tta_seed": int(args.tta_seed),
            "note": "mapping orientation identical to the published runs "
            "(cluster 0 -> down, cluster 1 -> up), so all batch metrics are "
            "unchanged; only the mapping-accuracy number and the reference-set "
            "descriptive statistics were updated.",
            "batches_unchanged_since": "2026-09-14",
        },
    }
    manifest["summary_rows"] = [
        row if r["split"] == "reference_200" else r for r in manifest["summary_rows"]
    ]
    (splits / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote: {splits/'reference_200.csv'}, {summary_path}, {splits/'manifest.json'}")


if __name__ == "__main__":
    main()
