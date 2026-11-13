# BTPC

Self-supervised P-wave first-motion polarity classification with Barlow Twins
and reference-based calibration. The model trains without polarity labels and
uses 200 manually labeled reference records to orient the learned classes.

## Current revised-manuscript primary experiment

Use [the SCEDC primary release](paper_artifacts/scedc_primary_20261113/README.md)
for the current training-seed-42 model, record lists, three target prediction
CSVs, independently recounted summaries and current reference-mapping evidence.

| Batch | Before-rejection accuracy | Coverage | Accepted-subset accuracy |
|---|---:|---:|---:|
| Training, 10,000 | 95.320% | 97.010% | 96.124111% |
| Validation, 100,000 | 95.116% | 96.942% | 95.969755% |
| Consensus, 100,000 | 99.294% | 98.512% | 99.550309% |

The current reference result is 192/200 = 96.00%. The consensus batch is a
secondary consistency check. This release covers the primary SCEDC experiment;
it does not claim a complete release of every supplementary or Ridgecrest run.

```bash
# Exact recount of the frozen results; Python standard library only.
python scripts/verify_paper_artifacts.py

# Independent inference on the 200 current reference records.
pip install -e .
python scripts/verify_paper_artifacts.py --reference-inference
```

See the package README for fresh target-inference commands, data acquisition,
SHA-256 identities, reference sampling and known reproducibility limits. The
historical target TTA RNG states were not saved; recounting the released CSVs
and running a new seeded inference realization are different operations.

## Historical files retained

The top-level [artifacts](artifacts/README.md) and [data_splits](data_splits/README.md)
directories retain the earlier public snapshot. They are **not the current
manuscript artifact bundle**. In particular, the old validation predictions were
produced by the historical `repro_v2_s42` model while the old seed42 checkpoint
was the native SCEDC model; those files must not be combined as a reproducible
current-paper run. No historical model or prediction file was deleted.

The earlier source and files are accessible at commit
[f5fdfde](https://github.com/suxieglitter/btpc-release/tree/f5fdfdec4de872171be95bb01e972e26fd63e9c2).
Use that commit to inspect the historical code. The root code at the current
commit supports the new package, whose code hashes are listed in its manifest.

## Data

Native input is the Ross et al. (2018) first-motion dataset distributed by
[SCEDC](https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5).
The complete native and derived consensus HDF5 files are not included. The
current exact row lists are under
`paper_artifacts/scedc_primary_20261113/data_splits/`, not the historical root
`data_splits/` directory. See the package README for what can be verified
without the full waveform inputs and what requires them.

## Citation

Pei, Y., and Ge, Z. BTPC: a Barlow Twins-based self-supervised P-wave
first-motion polarity clustering framework with weak reference-based
calibration. *Seismological Research Letters* (submitted).

Ross, Z. E., Meier, M.-A., and Hauksson, E. (2018). P wave arrival picking and
first-motion polarity determination with deep learning. *Journal of Geophysical
Research: Solid Earth*, 123(6), 5120–5129.
[DOI](https://doi.org/10.1029/2017JB015259).

MIT license; see LICENSE.
