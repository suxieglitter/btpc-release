"""Generate the committed data-split lists for the SCEDC experiment.

Writes the exact, seeded index lists to ``data_splits/`` so every batch in
the paper is reproducible:

- ``native_train_10k.csv``   Stage 1 training rows (seeded selection,
  identical to what ``btpc-train`` reconstructs from its config).
- ``native_valid_100k.csv``  Validation batch drawn from the remaining
  Up/Down records (training rows and reference rows excluded).
- ``reference_200.csv``      The 200 high-SNR reference records (100 up /
  100 down) used at prediction time for the cluster-to-polarity mapping.
- ``consensus_100k.csv``     Consistency-check subset of the consensus file,
  excluding any record that is part of the training batch or the references
  (matched by event id + station channel code).
- ``split_summary.csv``      Descriptive statistics per batch (class
  balance, event/station counts, SNR/magnitude quantiles).
- ``manifest.json``          Seeds, pool sizes and exclusion bookkeeping.

The reference bank itself (waveforms) is a run product and is not committed;
the reference list here is its committed 200-row subset.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from btpc.anchor_utils import ensure_anchor_bank, load_anchor_bank, sample_anchor_subset  # noqa: E402
from btpc.predict import get_stage1_selection_pools  # noqa: E402
from btpc.utils import checked_path, csv_text, save_json  # noqa: E402

SPLIT_CSV_COLUMNS = [
    "source_index",
    "evid",
    "sncl",
    "label",
    "label_name",
    "snr",
    "mag",
    "dist",
]

LABEL_NAMES = {0: "up", 1: "down", 2: "uncertain"}


def decode_sncls(values):
    decoded = []
    for value in np.asarray(values):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(value))
    return np.asarray(decoded, dtype=object)


def read_metadata(path, indices=None, keys=("Y", "snr", "evids", "sncls", "mag", "dist")):
    with h5py.File(str(path), "r") as handle:
        arrays = {}
        for key in keys:
            if key not in handle:
                arrays[key] = None
                continue
            values = handle[key][:]
            arrays[key] = values if indices is None else values[indices]
    if arrays.get("sncls") is not None:
        arrays["sncls"] = decode_sncls(arrays["sncls"])
    return arrays


def build_train_indices(native_path, snr_range, num_used, selection_seed):
    """Reproduce the seeded Stage 1 training selection on the native file."""
    selection_config = {
        "train_data_path": str(native_path),
        "data_label_key": "Y",
        "data_snr_key": "snr",
        "snr_range": [float(snr_range[0]), float(snr_range[1])],
        "num_used": int(num_used),
        "bino": True,
        "data_selection_seed": int(selection_seed),
    }
    train_indices, candidate_pool = get_stage1_selection_pools(selection_config, None)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(candidate_pool, dtype=np.int64)


def build_reference_subset(
    native_path,
    bank_path,
    manifest_path,
    anchor_snr_range,
    pool_per_label,
    build_seed,
    n_per_label,
    sample_seed,
):
    """Build (once) the high-SNR anchor bank from the native file and draw the
    balanced reference subset from it."""
    ensure_anchor_bank(
        bank_path=str(bank_path),
        manifest_path=str(manifest_path),
        data_path=str(native_path),
        snr_range=tuple(float(v) for v in anchor_snr_range),
        pool_per_label=int(pool_per_label),
        seed=int(build_seed),
    )
    bank = load_anchor_bank(str(bank_path))
    return sample_anchor_subset(bank, n_per_label=int(n_per_label), seed=int(sample_seed))


def build_valid_indices(native_path, snr_range, exclude_rows, size, seed):
    """Draw the validation batch from native Up/Down records, excluding the
    given rows (training batch and reference rows)."""
    with h5py.File(str(native_path), "r") as handle:
        labels = np.asarray(handle["Y"][:]).reshape(-1)
        snrs = np.asarray(handle["snr"][:]).reshape(-1)

    eligible = np.where(
        (labels != 2)
        & (snrs >= float(snr_range[0]))
        & (snrs < float(snr_range[1]))
    )[0]
    pool = np.setdiff1d(eligible, np.asarray(exclude_rows, dtype=np.int64), assume_unique=False)
    if pool.size < int(size):
        raise ValueError(
            f"Validation pool has {pool.size} rows, cannot draw {size} without exclusions."
        )
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(pool, size=int(size), replace=False)
    return np.sort(chosen.astype(np.int64))


def record_key_pairs(path, indices):
    """(evid, sncl) string keys identifying records across the two files."""
    with h5py.File(str(path), "r") as handle:
        evids = np.asarray(handle["evids"][:])[np.asarray(indices, dtype=np.int64)]
        sncls = decode_sncls(np.asarray(handle["sncls"][:])[np.asarray(indices, dtype=np.int64)])
    return {f"{int(evid)}|{sncl}" for evid, sncl in zip(evids, sncls)}


def build_consensus_indices(consensus_path, excluded_keys, size, seed):
    """Draw the consensus secondary-metric batch, skipping records that match
    the training batch or the references by (evid, sncl)."""
    with h5py.File(str(consensus_path), "r") as handle:
        labels = np.asarray(handle["Y"][:]).reshape(-1)
        evids = np.asarray(handle["evids"][:])
        sncls = decode_sncls(np.asarray(handle["sncls"][:]))

    keys = np.array(
        [f"{int(evid)}|{sncl}" for evid, sncl in zip(evids, sncls)], dtype=object
    )
    excluded_mask = np.isin(keys, list(excluded_keys)) if excluded_keys else np.zeros(len(keys), bool)
    eligible = np.where((labels != 2) & ~excluded_mask)[0]
    if eligible.size < int(size):
        raise ValueError(
            f"Consensus pool has {eligible.size} rows after exclusions, cannot draw {size}."
        )
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(eligible, size=int(size), replace=False)
    return np.sort(chosen.astype(np.int64)), int(excluded_mask.sum())


def summarize_split(name, labels, snrs, evids, sncls, mags, dists=None):
    labels = np.asarray(labels, dtype=np.int64)
    # float64 casts: np.percentile overflows on large float16 arrays (the mag
    # datasets are float16) when computing virtual indexes.
    snrs = np.asarray(snrs, dtype=np.float64)
    mags = np.asarray(mags, dtype=np.float64)
    row = {
        "split": name,
        "n": int(labels.size),
        "n_up": int(np.sum(labels == 0)),
        "n_down": int(np.sum(labels == 1)),
        "up_fraction": float(np.mean(labels == 0)) if labels.size else float("nan"),
        "n_events": int(np.unique(evids).size),
        "n_stations": int(np.unique([parse_station(s) for s in sncls]).size),
        "snr_mean": float(np.mean(snrs)),
        "snr_p10": float(np.percentile(snrs, 10)),
        "snr_p25": float(np.percentile(snrs, 25)),
        "snr_p50": float(np.percentile(snrs, 50)),
        "snr_p75": float(np.percentile(snrs, 75)),
        "snr_p90": float(np.percentile(snrs, 90)),
        "mag_p50": float(np.percentile(mags, 50)),
        "mag_mean": float(np.mean(mags)),
    }
    if dists is not None:
        dists = np.asarray(dists, dtype=np.float64)
        row["dist_p50"] = float(np.percentile(dists, 50))
    return row


def parse_station(sncl):
    parts = str(sncl).split(".")
    return parts[1] if len(parts) > 1 else str(sncl)


def write_split_csv(path, indices, metadata):
    labels = np.asarray(metadata["Y"], dtype=np.int64)
    snrs = np.asarray(metadata["snr"], dtype=float)
    evids = np.asarray(metadata["evids"])
    sncls = metadata["sncls"]
    mags = np.asarray(metadata["mag"], dtype=float)
    dists = np.asarray(metadata["dist"], dtype=float)
    rows = []
    for pos, source_index in enumerate(np.asarray(indices, dtype=np.int64)):
        rows.append(
            [
                int(source_index),
                int(evids[pos]),
                str(sncls[pos]),
                int(labels[pos]),
                LABEL_NAMES.get(int(labels[pos]), f"label_{int(labels[pos])}"),
                float(snrs[pos]),
                float(mags[pos]),
                float(dists[pos]),
            ]
        )
    Path(checked_path(str(path))).write_text(csv_text(SPLIT_CSV_COLUMNS, rows), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--native-path", required=True)
    parser.add_argument("--consensus-path", required=True)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data_splits"))
    parser.add_argument("--bank-dir", default=str(REPO_ROOT / "runs" / "native_v2"))
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=1000.0)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--valid-size", type=int, default=100000)
    parser.add_argument("--consensus-size", type=int, default=100000)
    parser.add_argument("--selection-seed", type=int, default=42,
                        help="Training selection seed; must match the training config.")
    parser.add_argument("--valid-seed", type=int, default=20260914)
    parser.add_argument("--consensus-seed", type=int, default=20260914)
    parser.add_argument("--anchor-snr-min", type=float, default=50.0)
    parser.add_argument("--anchor-pool-per-label", type=int, default=5000)
    parser.add_argument("--anchor-build-seed", type=int, default=42)
    parser.add_argument("--anchors-per-label", type=int, default=100)
    parser.add_argument("--anchor-sample-seed", type=int, default=20260404)
    args = parser.parse_args(argv)

    output_dir = Path(checked_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = Path(checked_path(args.bank_dir))
    bank_dir.mkdir(parents=True, exist_ok=True)
    snr_range = (float(args.snr_min), float(args.snr_max))

    print(f"[1/5] Training batch: seeded selection of {args.train_size} rows "
          f"(selection seed {args.selection_seed}).")
    train_indices, candidate_pool = build_train_indices(
        args.native_path, snr_range, args.train_size, args.selection_seed
    )
    train_meta = read_metadata(args.native_path, train_indices)
    write_split_csv(output_dir / "native_train_10k.csv", train_indices, train_meta)

    print(f"[2/5] Reference subset: bank from native SNR>={args.anchor_snr_min}, "
          f"{args.anchors_per_label}/label (build seed {args.anchor_build_seed}, "
          f"sample seed {args.anchor_sample_seed}).")
    bank_path = bank_dir / "anchor_bank_native.npz"
    subset = build_reference_subset(
        native_path=args.native_path,
        bank_path=bank_path,
        manifest_path=bank_dir / "anchor_bank_native_manifest.csv",
        anchor_snr_range=(args.anchor_snr_min, snr_range[1]),
        pool_per_label=args.anchor_pool_per_label,
        build_seed=args.anchor_build_seed,
        n_per_label=args.anchors_per_label,
        sample_seed=args.anchor_sample_seed,
    )
    ref_rows = np.asarray(subset["source_indices"], dtype=np.int64)
    ref_meta = read_metadata(args.native_path, ref_rows)
    write_split_csv(output_dir / "reference_200.csv", ref_rows, ref_meta)

    print(f"[3/5] Validation batch: {args.valid_size} rows from remaining native records "
          f"(seed {args.valid_seed}).")
    exclude_rows = np.concatenate([train_indices, ref_rows])
    valid_indices = build_valid_indices(
        args.native_path, snr_range, exclude_rows, args.valid_size, args.valid_seed
    )
    valid_meta = read_metadata(args.native_path, valid_indices)
    write_split_csv(output_dir / "native_valid_100k.csv", valid_indices, valid_meta)

    print(f"[4/5] Consensus batch: {args.consensus_size} rows (seed {args.consensus_seed}).")
    excluded_keys = record_key_pairs(args.native_path, exclude_rows)
    consensus_indices, n_excluded_records = build_consensus_indices(
        args.consensus_path, excluded_keys, args.consensus_size, args.consensus_seed
    )
    consensus_meta = read_metadata(args.consensus_path, consensus_indices)
    write_split_csv(output_dir / "consensus_100k.csv", consensus_indices, consensus_meta)

    print("[5/5] Descriptive statistics and manifest.")
    summary_rows = [
        summarize_split(
            "native_train_10k",
            train_meta["Y"], train_meta["snr"], train_meta["evids"],
            train_meta["sncls"], train_meta["mag"], train_meta["dist"],
        ),
        summarize_split(
            "native_valid_100k",
            valid_meta["Y"], valid_meta["snr"], valid_meta["evids"],
            valid_meta["sncls"], valid_meta["mag"], valid_meta["dist"],
        ),
        summarize_split(
            "reference_200",
            ref_meta["Y"], ref_meta["snr"], ref_meta["evids"],
            ref_meta["sncls"], ref_meta["mag"], ref_meta["dist"],
        ),
        summarize_split(
            "consensus_100k",
            consensus_meta["Y"], consensus_meta["snr"], consensus_meta["evids"],
            consensus_meta["sncls"], consensus_meta["mag"], consensus_meta["dist"],
        ),
    ]
    header = list(summary_rows[0].keys())
    Path(checked_path(str(output_dir / "split_summary.csv"))).write_text(
        csv_text(header, [[row[key] for key in header] for row in summary_rows]),
        encoding="utf-8",
    )

    with h5py.File(str(args.native_path), "r") as handle:
        native_labels = np.asarray(handle["Y"][:]).reshape(-1)
        native_snrs = np.asarray(handle["snr"][:]).reshape(-1)
    native_eligible = int(np.sum(
        (native_labels != 2) & (native_snrs >= snr_range[0]) & (native_snrs < snr_range[1])
    ))
    train_ref_overlap = int(np.intersect1d(train_indices, ref_rows).size)

    manifest = {
        "native_path": str(Path(args.native_path).resolve()),
        "consensus_path": str(Path(args.consensus_path).resolve()),
        "snr_range": list(snr_range),
        "train": {
            "size": int(train_indices.size),
            "selection": "seeded_2x_candidates_bino_filter_first_n (btpc.predict.get_stage1_selection_pools)",
            "selection_seed": int(args.selection_seed),
            "pool_note": "all native Up/Down records; consensus-set records are NOT excluded",
        },
        "reference": {
            "size": int(ref_rows.size),
            "per_label": int(args.anchors_per_label),
            "bank_path": str(bank_path),
            "bank_snr_min": float(args.anchor_snr_min),
            "bank_pool_per_label": int(args.anchor_pool_per_label),
            "bank_build_seed": int(args.anchor_build_seed),
            "subset_sample_seed": int(args.anchor_sample_seed),
            "overlap_with_train_rows": train_ref_overlap,
        },
        "valid": {
            "size": int(valid_indices.size),
            "seed": int(args.valid_seed),
            "excluded_rows": int(exclude_rows.size),
            "eligible_pool": native_eligible,
        },
        "consensus": {
            "size": int(consensus_indices.size),
            "seed": int(args.consensus_seed),
            "excluded_records_matching_train_or_refs": n_excluded_records,
            "match_key": "evid|sncl",
        },
        "summary_rows": summary_rows,
    }
    save_json(str(output_dir / "manifest.json"), manifest)

    for row in summary_rows:
        print(
            f"{row['split']}: n={row['n']} up={row['n_up']} down={row['n_down']} "
            f"events={row['n_events']} stations={row['n_stations']} snr_p50={row['snr_p50']:.2f}"
        )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
