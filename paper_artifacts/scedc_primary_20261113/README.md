# SCEDC primary experiment: manuscript artifact release

This directory contains the primary **native SCEDC, training seed 42** experiment
used by the revised BTPC manuscript. It includes the actual three target prediction
CSVs, their record lists, the encoder and classifier, original run configurations,
the reference bank and the current 200-record reference draw. The manifest records
file sizes, SHA-256 hashes and the local source of every copied artifact.

Public entry: [btpc-release SCEDC primary package](https://github.com/suxieglitter/btpc-release/tree/main/paper_artifacts/scedc_primary_20261113).

| Batch | Records | Accuracy before rejection | Coverage | Accepted-subset accuracy |
|---|---:|---:|---:|---:|
| Training | 10,000 | 95.320% | 97.010% | 96.124111% |
| Validation | 100,000 | 95.116% | 96.942% | 95.969755% |
| Consensus | 100,000 | 99.294% | 98.512% | 99.550309% |

These values are independently recounted from the released CSVs. Rejection is
`MC < 0.8 AND MCM < 0.05`; coverage uses the full batch denominator and accuracy
on accepted records uses only the retained records. The consensus batch is a
secondary consistency check selected by agreement between manual labels and
three classifiers, not an independent test population.

## Verify the reported numbers

From the repository root, this requires only the Python standard library:

```bash
python scripts/verify_paper_artifacts.py
```

The check verifies the manifest hashes, all 210,000 prediction-to-record identities,
manual labels, the cluster-to-polarity mapping and every AND-rejection outcome,
then recounts all three metrics and the stored reference result. No training,
waveform download or target inference is required for this calculation.

To independently rerun the 200-reference calculation with the released classifier:

```bash
pip install -e .
python scripts/verify_paper_artifacts.py --reference-inference
```

This reconstructs the exclusion mask and seeded draw from the released bank and
metadata, checks the selected identities against `data_splits/reference_200.csv`,
and uses the historical randomized draw order for inference. Selection seed and
NumPy TTA seed are both `20261004`. The verified result is **192/200 = 96.00%**,
with cluster 0 mapped to Down (1) and cluster 1 to Up (0). It also checks tensor
equality between the Stage-1 encoder and the frozen encoder inside Stage 2.
The release was checked on CPU with PyTorch 2.14.0+cu130 and NumPy 2.4.6.

## Reference identity and historical predictions

The original target summaries recorded reference agreements of 98.0% (training
and validation) and 97.5% (consensus), using the older reference sample seed
`20260404`. Those original summaries are retained unchanged under `provenance/`.
The current reference draw excludes training records and events, validation
records and consensus record keys. It has the **same global class orientation**.
Every released target row is checked against this orientation; changing the
reference draw therefore does not relabel any frozen target prediction.

Each new batch summary keeps `prediction_time_reference` and `current_reference`
separate. The 96% agreement is an actual rerun on the current reference sample,
not a replacement of the old summary number. Target CSVs are copied byte for byte
from the current manuscript results. Their historical target TTA RNG states were
not saved; exact fresh target-inference replay is not claimed. The model-to-CSV
provenance is the preserved run record and source hashes, not an independently
recovered historical RNG realization.

The validation list includes the eight replacements applied on 2026-10-14.
`provenance/validation_realized_replacements.json` compares the actual before/after
identities. `validation_final_selection.json` and `data_splits/manifest.json`
retain the recorded selection procedure, including its condition that the
reported quantities remain unchanged at their printed precision. This package
publishes that realized dataset; it does not reinterpret the replacements as a
new independently sampled validation experiment.

## Run a fresh prediction realization

The source HDF5 for training/validation is the Ross et al. (2018) SCEDC
first-motion dataset, available from the
[SCEDC data distribution](https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5).
The full source waveforms are not included here. The consensus HDF5 is the derived
`scsn_consensus_fm_CFM_DiTing_EQ.hdf5`; the record lists and predictions are provided,
but that full derived waveform file and its complete construction inputs are not
included. Fresh consensus inference therefore requires obtaining or reconstructing
that exact source file; the released row-level statistics remain verifiable
without it.

After obtaining the native file, run from the repository root:

```bash
python scripts/replay_paper_predictions.py \
  --batch validation \
  --data-path /path/to/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5 \
  --output-dir runs/fresh_scedc_validation \
  --tta-seed 20261113
```

Use `--batch training` with the same native source, or `--batch consensus` with
the exact derived consensus source. `--limit 16` gives a small diagnostic run.
The output directory must be empty and outside this frozen package. This command
rechecks the current reference draw, uses the released model and record order,
30 TTA views, maximum shift 1 sample, amplitude jitter 0.02, noise standard
deviation 0.015, batch size 256 and chunk size 8,192. Its newly seeded stochastic
realization can differ from the frozen manuscript CSV; the command writes its
own metrics, hashes and environment. Absolute paths in the original run
configuration are retained as provenance; `--data-path` supplies the reader's path.

## Scope and older release

This package covers the primary SCEDC experiment only. It does not claim to
contain complete Ridgecrest, baseline-model, seed-replicate or other supplementary
experiment artifacts. No fresh training was performed to build this release.

The earlier public `suxieglitter/btpc-release` snapshot at commit
[f5fdfde](https://github.com/suxieglitter/btpc-release/commit/f5fdfdec4de872171be95bb01e972e26fd63e9c2) contains older or mismatched prediction
artifacts and is not the artifact source for these revised manuscript numbers.
In particular, its validation predictions came from the historical
`repro_v2_s42` model, while its seed42 checkpoint was the native SCEDC model.
That historical snapshot remains available at the pinned commit.

Maintainers can rebuild this directory from the original local runs using
`python scripts/build_paper_artifacts.py`. Readers do not need those local runs.
Use a fixed Git commit when citing or downloading this package; file hashes in
`manifest.json` identify its contents independently of the branch name.
