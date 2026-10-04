# BTPC

Self-supervised P-wave first-motion polarity classification with Barlow
Twins and weak reference-based calibration.

BTPC trains a convolutional encoder with the Barlow Twins objective on
unlabeled P-wave windows, splits the learned feature space into two clusters
with spectral clustering, trains a lightweight classifier on the resulting
pseudo-labels, and maps the two clusters onto physical Up/Down polarities
with a small set of manually labeled reference waveforms (200 records in the
paper). No polarity label enters any training stage.

## Installation

```
pip install -e .
```

Requires Python >= 3.10; see `requirements.txt` for the package dependencies.

## Data

The pipeline reads SCEDC P-wave waveforms from an HDF5 file (waveforms,
binary polarity labels, SNR, event/station metadata). The primary
experiment uses the first-motion-polarity dataset of Ross et al. (2018),
distributed by the Southern California Earthquake Data Center (SCEDC):

https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5

Set `data_path` in `configs/native_v2.yaml` to the downloaded file. The
same dataset is also available programmatically through
[SeisBench](https://seisbench.readthedocs.io) as `Ross2018JGRFM`.
Underlying waveforms and earthquake catalogs: Southern California Earthquake
Data Center, https://doi.org/10.7909/C3WD3xH1.

The consistency-check batch additionally uses `scsn_consensus_fm_CFM_EQ.hdf5`,
a derived dataset (records on which the manual label and the CFM and EQPolar
predictions agree). It is not redistributed here because it embeds SCEDC
waveforms; its construction is described in the paper, and its role is
documented by the split list `data_splits/consensus_100k.csv`.

The exact row lists of every batch used in the paper (training 10k, validation 100k,
reference 200, consistency-check 100k, t-SNE subsample 10k) are committed
under `data_splits/` and can be regenerated with fixed seeds:

```
python scripts/make_event_splits.py \
    --native-path <scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5> \
    --consensus-path <scsn_consensus_fm_CFM_EQ.hdf5> \
    --output-dir data_splits
```

## Usage

```
# Stage 1: Barlow Twins representation learning + periodic filtering
btpc-train stage1 --config configs/native_v2.yaml

# Stage 2: pseudo-label classifier on the frozen encoder
btpc-train stage2 --stage1-dir <stage1_output_dir>

# Predict: test-time augmentation, MC/MCM scores, cluster-to-polarity mapping
btpc-predict --stage2-checkpoint <checkpoint> --data-path <hdf5>

# Evaluate predictions against manual labels
btpc-valid --stage2-checkpoint <checkpoint> --data-path <hdf5>
```

Run any command with `--help` for the full set of options.

## Released artifacts

Trained model checkpoints (the primary SCEDC run, the three seed
replicates, and the Ridgecrest application encoders), the reference-sample
banks, and the per-record prediction files of the labeled SCEDC batches
are included under `artifacts/`.

## Citation

Pei, Y., and Ge, Z. BTPC: a Barlow Twins-based self-supervised P-wave
first-motion polarity clustering framework with weak reference-based
calibration. *Seismological Research Letters* (submitted).

Dataset: Ross, Z. E., Meier, M.-A., and Hauksson, E. (2018). P wave
arrival picking and first-motion polarity determination with deep
learning. *Journal of Geophysical Research: Solid Earth*, 123(6),
5120–5129. https://doi.org/10.1029/2017JB015259

## License

MIT
