# Artifacts

Released run artifacts of the paper.

## Checkpoints (`checkpoints/`)

- `seed42/`–`seed45/`: Stage-1 encoder (`final_pwave_model.pth`) and
  Stage-2 pseudo-label classifier (`stage2_cluster_classifier.pth`) of the
  SCEDC experiment for training seeds 42–45 (42 = the primary run), each
  with its run configuration (`stage1_config.json`), summaries, and the
  final pseudo-label list.
- `ridgecrest/ridgecrest_encoder_snr{0,5,10,20}.pth`: self-supervised
  encoders retrained on the Ridgecrest subsets with SNR >= 0/5/10/20
  (SNR >= 10 is the paper's primary setting; the others are the
  training-SNR comparison).

## Reference banks (`reference_bank/`)

Anchor banks from which the 200 reference records are drawn at prediction
time, with manifests: `anchor_bank_native.npz` (SCEDC experiment),
`anchor_bank_scsn_paper.npz` (historical SCEDC records used in the
Ridgecrest application). The committed 200-row lists are
`data_splits/reference_200.csv` and the Ridgecrest manifest.

## Predictions (`predictions/`)

Per-record prediction files of the three labeled SCEDC batches
(`native_train10k`, `native_valid100k`, `consensus100k`), including the
per-record mean confidence (MC), cluster-center margin (MCM), and
rejection outcome, plus each batch's summary metrics.
