# Historical artifacts

This directory preserves artifacts from the public snapshot at commit
[f5fdfde](https://github.com/suxieglitter/btpc-release/tree/f5fdfdec4de872171be95bb01e972e26fd63e9c2).
It is not the current revised-manuscript primary experiment package.
Use [the current release](../paper_artifacts/scedc_primary_20261113/README.md)
for corresponding models, record lists, predictions, reference mapping and
independently recounted metrics.

The old `predictions/native_valid100k` CSV and summary record the historical
`repro_v2_s42` model (94.902% raw accuracy, 95.459% coverage,
96.214081% accepted accuracy and 99.5% reference agreement). The old
`checkpoints/seed42` classifier belongs to the native SCEDC run. These are
different model realizations and must not be paired as one current-paper run.
The reference list and the target-inference settings also predate later
manuscript changes.

Existing checkpoints, reference banks and prediction files are retained
unchanged for historical inspection. Their presence here does not establish
that every supplementary, seed-replicate or Ridgecrest result in the current
manuscript is fully packaged or independently reproducible.
