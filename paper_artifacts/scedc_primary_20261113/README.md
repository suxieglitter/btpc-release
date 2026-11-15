# SCEDC primary experiment

This package contains the seed-42 encoder and classifier, experiment settings,
record lists, and per-record predictions for the BTPC SCEDC experiment. It
includes the 200 labeled reference records used to map clusters to Up and Down.

| Batch | Records | Accuracy before rejection | Coverage | Accuracy on accepted records |
|---|---:|---:|---:|---:|
| Training | 10,000 | 95.320% | 97.010% | 96.124% |
| Validation | 100,000 | 95.116% | 96.942% | 95.970% |
| Consensus | 100,000 | 99.294% | 98.512% | 99.550% |

A record is rejected when `MC < 0.8` and `MCM < 0.05`. Coverage is the fraction
of all records retained. The consensus batch checks agreement on records whose
manual labels match CFM, DiTingMotion, and EQPolar.

## Verify the results

Run from the repository root. This command checks file hashes, record identities,
labels, and rejection decisions, and recomputes the metrics from the saved CSVs.
It uses the Python standard library.

```bash
python scripts/verify_paper_artifacts.py
```

To run the classifier on the 200 reference records and verify the polarity mapping,
install the package and run the reference check. Reference agreement is 192/200
(96.00%).

```bash
python -m pip install -e .
python scripts/verify_paper_artifacts.py --reference-inference
```

## Run inference on waveforms

Use the [SCEDC first-motion waveform file][waveforms] for the training and
validation batches. The consensus record list and predictions are included.
Waveform inference on that batch requires the derived
`scsn_consensus_fm_CFM_DiTing_EQ.hdf5` file.

The [quick start](../../README.md#quick-start) runs inference on a small validation
subset. Remove `--limit 16` for the full batch. Select the training batch with
`--batch training` or supply the derived file for `--batch consensus`. Use a new
or empty output directory. New seeded TTA runs can yield different predictions
from the saved CSVs used to report the paper's results.

[waveforms]: https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5
