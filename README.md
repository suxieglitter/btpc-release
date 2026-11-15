# BTPC

BTPC is a self-supervised classifier for P-wave first-motion polarity based on
Barlow Twins. A small set of labeled reference waveforms maps the two learned
classes to Up and Down.

## Quick start

Install with Python 3.10 or later. We have tested this release only on Linux.

```bash
python -m pip install -e .
```

Download the [SCEDC first-motion waveform file][waveforms] and set `--data-path`
to its location. The example runs inference on 16 validation records and writes
`predictions.csv` and `summary.json` to `runs/example`.

```bash
python scripts/replay_paper_predictions.py \
  --batch validation \
  --data-path /path/to/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5 \
  --output-dir runs/example --limit 16
```

## Predict your own waveforms

The example below applies the released SCEDC model to your waveforms without
retraining. Save vertical-component traces in a NumPy `.npz` file under the key
`waveforms`, with shape `(N, 600)`. Each trace must be sampled at 100 Hz, with
the P arrival at sample index 300 (zero-based), and use an upward-positive
polarity convention. Supply the full six-second traces; the example detrends,
crops the 32-sample P window, and normalizes it internally. Manual polarity
labels and event/station metadata are not required.

```python
import numpy as np

# waveforms is your array of N aligned, six-second vertical-component traces.
np.savez("my_waveforms.npz", waveforms=waveforms.astype(np.float32))
```

From the repository root, after installation:

```bash
python examples/predict_waveforms.py \
  --input my_waveforms.npz \
  --output-dir runs/my_predictions
```

Use a new or empty output directory. In `predictions.csv`, `input_row` identifies
the zero-based input row and `final_label_name` is `Up`, `Down`, or `Uncertain`.
The example uses 30 TTA views and rejects a record only when both
`MC < 0.8` and `MCM < 0.05`, matching the SCEDC paper example above.
The separate `btpc-predict` command uses vote-consistency/margin rejection and
different TTA defaults; use the examples here for the paper's prediction settings.

## Paper artifacts

The [SCEDC experiment package](paper_artifacts/scedc_primary_20261113/README.md)
includes the trained encoder and classifier, record lists, and per-record
predictions for the training, validation, and consensus batches. It also includes
200 reference records and a command to check the reported results.

## Citation

Pei, Y., and Ge, Z. *BTPC: A Barlow Twins-based Self-Supervised P-Wave
First-Motion Polarity Clustering Framework with Weak Reference-Based Calibration*.

The SCEDC waveform dataset is described by [Ross et al. (2018)][ross].

BTPC is distributed under the [MIT license](LICENSE).

[waveforms]: https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5
[ross]: https://doi.org/10.1029/2017JB015251
