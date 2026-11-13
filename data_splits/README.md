# Historical data splits

These root-level CSVs and manifest are retained from the public snapshot at
[f5fdfde](https://github.com/suxieglitter/btpc-release/tree/f5fdfdec4de872171be95bb01e972e26fd63e9c2).
They are not the current revised-manuscript row lists. In particular, this
directory's reference draw uses seed 20260404 and predates the independent
reference redraw, and its validation list predates the eight replacements
applied on 2026-10-14.

Use the current package's
[data_splits directory](../paper_artifacts/scedc_primary_20261113/data_splits/)
and [reproduction instructions](../paper_artifacts/scedc_primary_20261113/README.md).
Its current reference sample seed is 20261004. Exact CSV identities and
selection/replacement provenance are included in the package manifest and
recorded source manifest.

The historical `tsne_subsample_10k.csv` is retained here. The new primary package
does not claim to republish or independently verify the full t-SNE experiment.
Reconstructing old split files requires checking out the historical code at the
commit above; current root scripts follow the current selection rules.
