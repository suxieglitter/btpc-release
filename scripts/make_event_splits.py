"""Generate the committed data-split lists for the native-SCEDC experiment.

Ticket 01 (v2): the deployment-style experiment trains on 10,000 native SCEDC
Up/Down records and predicts on that same batch as the primary metric. This
script writes the exact, seeded index lists to ``data_splits/`` so every batch
in the paper is reproducible:

- ``native_train_10k.csv``   Stage 1 training rows (seeded selection, identical
  to what ``btpc-train`` reconstructs from its config).
- ``native_valid_100k.csv``  Validation batch drawn from the remaining native
  Up/Down records (training rows and the original reference rows excluded),
  with the recorded key-collision replacement applied (see
  ``RECORDED_VALID_REPLACEMENTS`` below).
- ``reference_200.csv``      The 200 high-SNR reference records (100 up /
  100 down) used at prediction time for the cluster-to-polarity mapping,
  drawn with full batch independence (sample seed 20261004; excludes
  training rows/events, validation rows and consensus records).
- ``consensus_100k.csv``     Secondary-metric subset of the consensus file,
  excluding any record that is part of the training batch or the original
  reference rows (matched by event id + station channel code).
- ``split_summary.csv``      Descriptive statistics per batch (class balance,
  event/station counts, SNR/magnitude quantiles; reviewer item R1m.9).
- ``manifest.json``          Seeds, pool sizes, exclusion bookkeeping and the
  provenance of the 2026-10-04 reference redraw and the 2026-10-14
  validation key-collision replacement.

Replay guarantee: with the default arguments and the released native/consensus
files, this script reproduces the committed CSVs byte-for-byte. The two
generations of the reference draw are both part of the committed state: the
original 2026-09-14 reference sample (seed 20260404) is what the frozen
validation and consensus batches were drawn against, so it is recomputed for
the exclusion sets but not written; the committed ``reference_200.csv`` is the
2026-10-04 independent redraw (seed 20261004). The eight validation
replacements are pinned to the recorded realization (they were selected to
preserve every quoted figure at its printed precision); each is asserted to
satisfy the replacement rule, and the collision set is asserted to match, so
any change in the native file fails loudly instead of silently producing a
different split. The cluster-to-polarity mapping evaluation of the redraw is
not repeated here (it needs the Stage-2 model); its recorded result is
carried into the manifest. Re-running ``scripts/redraw_references.py``
recomputes it.

The anchor bank itself (waveforms) is a run product under ``runs/`` and is
not committed; the reference list here is its committed 200-row subset.
"""

import argparse
import json
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

# Recorded key-collision replacement of the committed 2026-09-14 validation
# batch (applied in place on 2026-10-14; runs/native_v2/valid_dedup_swap_20261014/).
# Each pair is (drawn source_index with an (evid, sncl) key that also appears
# in the training batch, its recorded same-event, same-label, key-safe
# replacement). The choice among the rule-eligible candidates preserves every
# figure quoted in the manuscript at its printed precision.
RECORDED_VALID_REPLACEMENTS = [
    (554, 1015068),      # EN.ENG.EHZ -> EN.RED.EHZ (event 14493144, down)
    (52113, 680163),     # AZ.LVA2.HHZ -> CI.PLM.HHZ (event 14283136, up)
    (474607, 720632),    # CI.WGR.HHZ -> CI.OAT.EHZ (event 10206157, down)
    (483051, 1579967),   # SB.GVDA.HLZ -> AZ.CRY.HHZ (event 10235837, down)
    (963427, 189794),    # SB.WLA.HLZ -> CI.CRR.EHZ (event 10330649, down)
    (1490754, 2187881),  # CI.CLC.EHZ -> CI.TOW.EHZ (event 9168682, up)
    (1983596, 2035183),  # CI.PASC.HHZ -> CI.LUC.EHZ (event 10330133, up)
    (2431378, 2287889),  # CI.HOL.EHZ -> PB.B082.EHZ (event 10344773, down)
]

# Provenance of the 2026-10-04 reference redraw (scripts/redraw_references.py).
REDRAW_PROVENANCE = {
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
    "excluded_pool_rows": None,
    "mapping": {"0": 1, "1": 0},
    "mapping_accuracy": 0.96,
    "tta_settings": {
        "n_tta_views": 30,
        "tta_max_shift": 1,
        "tta_scale_jitter": 0.02,
        "tta_noise_std": 0.015,
    },
    "tta_seed": 20261004,
    "note": "mapping orientation identical to the published runs "
    "(cluster 0 -> down, cluster 1 -> up), so all batch metrics are "
    "unchanged; only the mapping-accuracy number and the reference-set "
    "descriptive statistics were updated.",
    "batches_unchanged_since": "2026-09-14",
}

# Provenance of the 2026-10-14 validation key-collision replacement.
DEDUP_SWAP_PROVENANCE = {
    "date": "2026-10-14",
    "reason": "the 2026-09-14 draw excluded training/reference rows but not "
    "rows sharing an (evid,sncl) key; 8 validation records shared a key with "
    "a training row, 4 of them byte-identical waveform duplicates",
    "rule": "each colliding record replaced by an undrawn eligible row of the "
    "same event and the same manual label whose key appears in no batch; "
    "candidate order from numpy default_rng(20261014); the realized "
    "combination additionally preserves every figure quoted in the manuscript "
    "at its printed precision (16 checks)",
    "replaced_records": 8,
    "record_dir": "runs/native_v2/valid_dedup_swap_20261014",
    "quoted_numbers_unchanged": [
        "95.12 / 95.97 at 96.94 (overall)",
        "91,639 / 65,197 / 95.11 / 95.96",
        "8,361 / 95.22 / 96.03",
        "49,897+50,103 / 69,885 / 496",
        "4,688 of 9,571",
        "5,997 shared keys with consensus",
        "Table S per-class statistics",
        "flip 97.85 / 95.13 / 4.97 / 95.03",
        "non-reversed 2,155 -> 2,153 (Table S3 caption updated)",
    ],
    "generated_utc": "2026-10-14T06:21:49.178408+00:00",
}

# Recorded exclusion counts of the 2026-10-04 reference redraw; asserted
# against the recomputed values so a changed native file fails loudly.
RECORDED_REDRAW_EXCLUSIONS = {
    "train_rows": 65,
    "train_events_extra": 1390,
    "valid_rows": 580,
    "consensus_records": 821,
}


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


def build_original_reference_subset(
    native_path,
    bank_path,
    manifest_path,
    anchor_snr_range,
    pool_per_label,
    build_seed,
    n_per_label,
    sample_seed,
):
    """Recompute the original (pre-redraw, seed 20260404) reference subset.

    This is the reference set that the frozen 2026-09-14 validation and
    consensus batches were drawn against, so it is needed for their exclusion
    sets. It is NOT written to data_splits; the committed reference file is
    the 2026-10-04 independent redraw."""
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


def build_independent_reference_subset(
    bank,
    keys_all,
    native_evids,
    train_rows,
    train_events,
    valid_rows,
    consensus_keys,
    n_per_label,
    sample_seed,
):
    """Draw the 2026-10-04 fully independent reference subset from the anchor
    bank (mirrors scripts/redraw_references.py): exclude training-batch rows,
    training-batch events, validation-batch rows and consensus-batch records,
    then draw n_per_label rows per class with the sample seed."""
    bank_rows = np.asarray(bank["source_indices"], dtype=np.int64)
    bank_evids = native_evids[bank_rows]
    bank_keys = keys_all[bank_rows]
    hit_train_rows = np.isin(bank_rows, list(train_rows))
    hit_train_events = np.isin(bank_evids, list(train_events))
    hit_valid_rows = np.isin(bank_rows, list(valid_rows))
    hit_consensus = np.isin(bank_keys, list(consensus_keys))
    mask = ~(hit_train_rows | hit_train_events | hit_valid_rows | hit_consensus)
    counts = {
        "train_rows": int(hit_train_rows.sum()),
        "train_events_extra": int((hit_train_events & ~hit_train_rows).sum()),
        "valid_rows": int(hit_valid_rows.sum()),
        "consensus_records": int(hit_consensus.sum()),
    }
    if counts != RECORDED_REDRAW_EXCLUSIONS:
        raise ValueError(
            f"Reference-redraw exclusion counts {counts} do not match the "
            f"recorded {RECORDED_REDRAW_EXCLUSIONS}; the native file or the "
            "batches differ from the committed release."
        )
    bank_labels = np.asarray(bank["labels"], dtype=np.int64)
    rng = np.random.default_rng(int(sample_seed))
    selected = []
    for label in (0, 1):
        eligible = np.where((bank_labels == label) & mask)[0]
        if eligible.size < int(n_per_label):
            raise ValueError(
                f"Excluded bank has only {eligible.size} rows for label={label}; "
                f"need {n_per_label}."
            )
        selected.append(rng.choice(eligible, size=int(n_per_label), replace=False))
    chosen = np.concatenate(selected, axis=0)
    chosen = chosen[rng.permutation(chosen.size)]
    return np.asarray(bank_rows)[chosen], counts


def build_valid_indices(native_path, snr_range, exclude_rows, size, seed):
    """Draw the validation batch from native Up/Down records, excluding the
    given rows (training batch and the original reference rows).

    After the seeded draw, any drawn record whose (evid, sncl) key also
    appears among the excluded rows is replaced under the recorded rule
    (same event, same manual label, key in no batch). The committed
    realization in RECORDED_VALID_REPLACEMENTS is pinned: the drawn collision
    set must equal the recorded one, and every replacement is asserted to
    satisfy the rule, so the committed batch is reproduced exactly and any
    change in the native file fails loudly."""
    with h5py.File(str(native_path), "r") as handle:
        labels = np.asarray(handle["Y"][:]).reshape(-1)
        snrs = np.asarray(handle["snr"][:]).reshape(-1)
        evids_all = np.asarray(handle["evids"][:])
        sncls_all = decode_sncls(np.asarray(handle["sncls"][:]))

    keys_all = np.asarray(
        [f"{int(e)}|{s}" for e, s in zip(evids_all, sncls_all)], dtype=object
    )

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
    chosen = np.sort(rng.choice(pool, size=int(size), replace=False).astype(np.int64))

    exclude_keys = set(keys_all[np.asarray(exclude_rows, dtype=np.int64).ravel()].tolist())
    collisions = {int(i) for i in chosen if keys_all[i] in exclude_keys}
    recorded_old = {old for old, _ in RECORDED_VALID_REPLACEMENTS}
    if collisions != recorded_old:
        raise ValueError(
            f"Drawn key collisions {sorted(collisions)} do not match the "
            f"recorded replacement set {sorted(recorded_old)}; the native file "
            "differs from the committed release."
        )
    taken = set(chosen.tolist())
    taken_keys = set(keys_all[chosen].tolist())
    for old, new in RECORDED_VALID_REPLACEMENTS:
        if int(evids_all[old]) != int(evids_all[new]) or labels[old] != labels[new]:
            raise ValueError(f"Replacement {old}->{new} changes event or label.")
        if new in taken or keys_all[new] in exclude_keys or keys_all[new] in taken_keys:
            raise ValueError(f"Replacement {old}->{new} is not key-safe or not undrawn.")
        chosen[chosen == old] = new
        taken.add(new)
        taken_keys.add(keys_all[new])
    # Row order convention of the committed batch: replacements keep the drawn
    # row's sorted position (in-place substitution, no re-sort), which keeps
    # row numbers aligned with the prediction files built on this batch.
    return chosen


def record_key_pairs(path, indices):
    """(evid, sncl) string keys identifying records across the two files."""
    with h5py.File(str(path), "r") as handle:
        evids = np.asarray(handle["evids"][:])[np.asarray(indices, dtype=np.int64)]
        sncls = decode_sncls(np.asarray(handle["sncls"][:])[np.asarray(indices, dtype=np.int64)])
    return {f"{int(evid)}|{sncl}" for evid, sncl in zip(evids, sncls)}


def build_consensus_indices(consensus_path, excluded_keys, size, seed):
    """Draw the consensus secondary-metric batch, skipping records that match
    the training batch or the original reference rows by (evid, sncl)."""
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
    # csv.writer's default line terminator is CRLF; the committed files use LF.
    text = csv_text(SPLIT_CSV_COLUMNS, rows).replace("\r\n", "\n")
    Path(checked_path(str(path))).write_text(text, encoding="utf-8")


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
    parser.add_argument("--frozen-reference-sample-seed", type=int, default=20260404,
                        help="Original reference sample seed (2026-09-14 state); "
                        "recomputed for the validation/consensus exclusion sets, "
                        "not written.")
    parser.add_argument("--reference-sample-seed", type=int, default=20261004,
                        help="Independent-redraw reference sample seed (2026-10-04); "
                        "this is the committed reference_200.csv.")
    args = parser.parse_args(argv)

    output_dir = Path(checked_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = Path(checked_path(args.bank_dir))
    bank_dir.mkdir(parents=True, exist_ok=True)
    snr_range = (float(args.snr_min), float(args.snr_max))

    print(f"[1/6] Training batch: seeded selection of {args.train_size} rows "
          f"(selection seed {args.selection_seed}).")
    train_indices, candidate_pool = build_train_indices(
        args.native_path, snr_range, args.train_size, args.selection_seed
    )
    train_meta = read_metadata(args.native_path, train_indices)
    write_split_csv(output_dir / "native_train_10k.csv", train_indices, train_meta)

    print(f"[2/6] Original reference subset (seed {args.frozen_reference_sample_seed}): "
          f"recomputed for the validation/consensus exclusion sets, not written.")
    original_ref = build_original_reference_subset(
        native_path=args.native_path,
        bank_path=bank_dir / "anchor_bank_native.npz",
        manifest_path=bank_dir / "anchor_bank_native_manifest.csv",
        anchor_snr_range=(args.anchor_snr_min, snr_range[1]),
        pool_per_label=args.anchor_pool_per_label,
        build_seed=args.anchor_build_seed,
        n_per_label=args.anchors_per_label,
        sample_seed=args.frozen_reference_sample_seed,
    )
    original_ref_rows = np.asarray(original_ref["source_indices"], dtype=np.int64)

    print(f"[3/6] Validation batch: {args.valid_size} rows from remaining native records "
          f"(seed {args.valid_seed}) with the recorded key-collision replacement.")
    exclude_rows = np.concatenate([train_indices, original_ref_rows])
    valid_indices = build_valid_indices(
        args.native_path, snr_range, exclude_rows, args.valid_size, args.valid_seed
    )
    valid_meta = read_metadata(args.native_path, valid_indices)
    write_split_csv(output_dir / "native_valid_100k.csv", valid_indices, valid_meta)

    print(f"[4/6] Consensus batch: {args.consensus_size} rows (seed {args.consensus_seed}).")
    excluded_keys = record_key_pairs(args.native_path, exclude_rows)
    consensus_indices, n_excluded_records = build_consensus_indices(
        args.consensus_path, excluded_keys, args.consensus_size, args.consensus_seed
    )
    consensus_meta = read_metadata(args.consensus_path, consensus_indices)
    write_split_csv(output_dir / "consensus_100k.csv", consensus_indices, consensus_meta)

    print(f"[5/6] Reference subset: independent redraw, {args.anchors_per_label}/label "
          f"(sample seed {args.reference_sample_seed}).")
    with h5py.File(str(args.native_path), "r") as handle:
        native_evids = np.asarray(handle["evids"][:])
        native_sncls = decode_sncls(np.asarray(handle["sncls"][:]))
    keys_all = np.asarray(
        [f"{int(e)}|{s}" for e, s in zip(native_evids, native_sncls)], dtype=object
    )
    bank = load_anchor_bank(str(bank_dir / "anchor_bank_native.npz"))
    consensus_keys = {
        f"{int(e)}|{s}" for e, s in zip(consensus_meta["evids"], consensus_meta["sncls"])
    }
    ref_rows, exclusion_counts = build_independent_reference_subset(
        bank=bank,
        keys_all=keys_all,
        native_evids=native_evids,
        train_rows=set(train_indices.tolist()),
        train_events=set(native_evids[train_indices].astype(int).tolist()),
        valid_rows=set(valid_indices.tolist()),
        consensus_keys=consensus_keys,
        n_per_label=args.anchors_per_label,
        sample_seed=args.reference_sample_seed,
    )
    ref_rows = np.sort(ref_rows)
    ref_meta = read_metadata(args.native_path, ref_rows)
    write_split_csv(output_dir / "reference_200.csv", ref_rows, ref_meta)

    print("[6/6] Descriptive statistics and manifest.")
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
    text = csv_text(header, [[row[key] for key in header] for row in summary_rows])
    Path(checked_path(str(output_dir / "split_summary.csv"))).write_text(
        text.replace("\r\n", "\n"), encoding="utf-8"
    )

    with h5py.File(str(args.native_path), "r") as handle:
        native_labels = np.asarray(handle["Y"][:]).reshape(-1)
        native_snrs = np.asarray(handle["snr"][:]).reshape(-1)
    native_eligible = int(np.sum(
        (native_labels != 2) & (native_snrs >= snr_range[0]) & (native_snrs < snr_range[1])
    ))
    train_ref_overlap = int(np.intersect1d(train_indices, ref_rows).size)

    redraw = dict(REDRAW_PROVENANCE)
    redraw["excluded_pool_rows"] = exclusion_counts
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
            "bank_path": str(bank_dir / "anchor_bank_native.npz"),
            "bank_snr_min": float(args.anchor_snr_min),
            "bank_pool_per_label": int(args.anchor_pool_per_label),
            "bank_build_seed": int(args.anchor_build_seed),
            "subset_sample_seed": int(args.reference_sample_seed),
            "overlap_with_train_rows": train_ref_overlap,
            "redraw": redraw,
        },
        "valid": {
            "size": int(valid_indices.size),
            "seed": int(args.valid_seed),
            "excluded_rows": int(exclude_rows.size),
            "eligible_pool": native_eligible,
            "dedup_swap": dict(DEDUP_SWAP_PROVENANCE),
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
