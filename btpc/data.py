"""Datasets, waveform preprocessing and augmentation, and HDF5 loading.

Two source layouts are supported. The SCSN polarity files are HDF5 with three
parallel datasets: ``X`` (waveforms, ``(n, 600)`` float), ``Y`` (labels,
0=up / 1=down / 2=uncertain) and ``snr``. The P arrival sits at sample index
300 with a 100 Hz sampling rate. The unlabeled Ridgecrest files carry a
``phasenet`` group with ``waveforms`` / ``snr`` / ``record_id`` and a P
arrival at sample index 1000; they are used for self-supervised training
without labels.
"""

import os
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
from scipy import signal
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .utils import resolve_data_path, runs_from_indices

DEFAULT_WAVEFORM_KEY = "X"
DEFAULT_LABEL_KEY = "Y"
DEFAULT_SNR_KEY = "snr"
DEFAULT_P_ARRIVAL_INDEX = 300
RIDGECREST_P_ARRIVAL_INDEX = 1000
DEFAULT_SAMPLING_RATE = 100.0


def require_hdf5_keys(handle, keys: Sequence[str], data_path: str):
    missing = [key for key in keys if key not in handle]
    if missing:
        available_keys = ", ".join(sorted(handle.keys()))
        raise KeyError(
            f"Missing dataset key(s) {missing} in {data_path}. "
            f"Available keys: {available_keys}"
        )


def read_scsn_rows(
    data_path: str,
    indices,
    waveform_key: str = DEFAULT_WAVEFORM_KEY,
    label_key: str = DEFAULT_LABEL_KEY,
    snr_key: str = DEFAULT_SNR_KEY,
    extra_keys: Sequence[str] = (),
):
    """Read the given rows from an SCSN-style HDF5 file in the caller's order.

    Returns ``(waveforms, labels, snrs, extras)`` where ``extras`` maps each
    key of ``extra_keys`` to its array, or to ``None`` when the file does not
    contain that dataset.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("No rows were selected to read.")

    runs, _, inv_order = runs_from_indices(indices)
    optional_present = []
    with h5py.File(data_path, "r") as handle:
        require_hdf5_keys(handle, (waveform_key, label_key, snr_key), data_path)
        optional_present = [key for key in extra_keys if key in handle]
        names = [waveform_key, label_key, snr_key] + optional_present
        datasets = {name: handle[name] for name in names}
        buffers = {name: [] for name in names}
        for start, end in runs:
            for name in names:
                buffers[name].append(datasets[name][start:end])

    arrays = {name: np.concatenate(buffers[name], axis=0)[inv_order] for name in names}
    extras = {key: arrays.pop(key) for key in optional_present}
    extras.update({key: None for key in extra_keys if key not in optional_present})
    return arrays[waveform_key], arrays[label_key], arrays[snr_key], extras


class SeismicPolarityDataset(Dataset):
    """P-wave polarity dataset with random-shift / noise / scale augmentation.

    ``__getitem__`` returns two augmented views, the clean (unaugmented)
    window, the polarity label and the SNR. Windows are cropped around the P
    arrival and normalized according to ``norm_mod`` (``"max"``, ``"rms"`` or
    ``"bn"``).
    """

    def __init__(
        self,
        waveforms,
        polarities,
        snrs,
        p_arrival_times,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        shift=0,
        p_window=0.5,
        apply_augmentation=True,
        aug_shift=1,
        aug_noise_std_range=(0.05, 0.2),
        aug_scale_range=(0.8, 1.2),
        norm_mod="max",
    ):
        self.waveforms = np.asarray(waveforms, dtype=np.float32)
        self.polarities = np.asarray(polarities, dtype=np.int64)
        self.snrs = np.asarray(snrs, dtype=np.float32)
        self.p_arrival_times = np.asarray(p_arrival_times, dtype=np.int64)
        self.sampling_rate = float(sampling_rate)
        self.shift = int(shift)
        self.p_window = float(p_window)
        self.apply_augmentation = bool(apply_augmentation)
        self.norm_mod = str(norm_mod)

        self.n_samples = len(self.waveforms)
        self.window_pts = int(self.p_window * self.sampling_rate)
        self.half_win = self.window_pts // 2
        self.time_shift_range = int(aug_shift)
        self.noise_std_range = tuple(float(v) for v in aug_noise_std_range)
        self.scale_range = tuple(float(v) for v in aug_scale_range)

    def __len__(self):
        return self.n_samples

    def normalize(self, waveform):
        return (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-8)

    def normalize_by_rms(self, waveform, noise_template):
        rms = np.sqrt(np.mean(noise_template ** 2))
        return waveform / rms if rms > 1e-6 else waveform

    def normalize_by_max(self, waveform):
        pos_max = np.max(np.abs(waveform))
        return waveform / pos_max if pos_max > 1e-6 else waveform

    def _preprocess_waveform(self, waveform, p_arrival_idx):
        waveform = signal.detrend(waveform)
        start_idx = max(0, p_arrival_idx - self.half_win + self.shift)
        end_idx = min(len(waveform), p_arrival_idx + self.half_win + self.shift)
        p_wave = waveform[start_idx:end_idx]

        noise_win_len = int(0.5 * self.sampling_rate)
        noise_start = max(0, p_arrival_idx - 10 - noise_win_len)
        noise_end = max(noise_start + 1, p_arrival_idx - 10)
        noise_template = waveform[noise_start:noise_end]

        if self.norm_mod == "rms":
            p_wave = self.normalize_by_rms(p_wave, noise_template)
        elif self.norm_mod == "max":
            p_wave = self.normalize_by_max(p_wave)
        elif self.norm_mod == "bn":
            p_wave = self.normalize(p_wave)

        return np.asarray(p_wave, dtype=np.float32)

    def _augment_waveform(self, waveform):
        if not self.apply_augmentation:
            return waveform

        augmented = waveform.copy()
        shift_random = np.random.randint(-self.time_shift_range, self.time_shift_range + 1)
        if shift_random > 0:
            augmented = np.concatenate([np.zeros(shift_random), augmented[:-shift_random]])
        elif shift_random < 0:
            augmented = np.concatenate([augmented[-shift_random:], np.zeros(-shift_random)])

        noise_std = np.random.uniform(*self.noise_std_range)
        augmented = augmented + np.random.normal(0, noise_std, augmented.shape)

        scale = np.random.uniform(*self.scale_range)
        augmented = augmented * scale

        if len(augmented) > 6:
            mask_len = np.random.randint(1, min(5, len(augmented) - 1))
            low = max(0, min(self.half_win - self.shift + 3, len(augmented) - mask_len))
            high = max(low + 1, len(augmented) - mask_len + 1)
            mask_start = np.random.randint(low, high)
            augmented[mask_start : mask_start + mask_len] = 0

        return np.asarray(augmented, dtype=np.float32)

    def get_clean_waveform(self, idx: int) -> np.ndarray:
        waveform = self.waveforms[idx]
        p_arrival_idx = int(self.p_arrival_times[idx])
        return self._preprocess_waveform(waveform, p_arrival_idx)

    def get_clean_waveforms(self, indices) -> np.ndarray:
        clean = [self.get_clean_waveform(int(idx)) for idx in np.asarray(indices, dtype=int)]
        return np.asarray(clean, dtype=np.float32)

    def __getitem__(self, idx):
        clean_wave = self.get_clean_waveform(int(idx))
        view1 = self._augment_waveform(clean_wave)
        view2 = self._augment_waveform(clean_wave)
        polarity = torch.tensor(int(self.polarities[idx]), dtype=torch.long)
        snr = torch.tensor(float(self.snrs[idx]), dtype=torch.float32)
        return (
            torch.FloatTensor(view1).unsqueeze(0),
            torch.FloatTensor(view2).unsqueeze(0),
            torch.FloatTensor(clean_wave).unsqueeze(0),
            polarity,
            snr,
        )


class PseudoLabelDataset(Dataset):
    """Clean-waveform dataset carrying Stage 2 pseudo labels and weights."""

    def __init__(self, base_dataset: SeismicPolarityDataset, indices, pseudo_labels, sample_weights=None):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)
        if sample_weights is None:
            sample_weights = np.ones(len(self.indices), dtype=np.float32)
        self.sample_weights = np.asarray(sample_weights, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_index = int(self.indices[idx])
        clean_wave = self.base_dataset.get_clean_waveform(sample_index)
        return (
            torch.FloatTensor(clean_wave).unsqueeze(0),
            torch.tensor(int(self.pseudo_labels[idx]), dtype=torch.long),
            torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32),
            torch.tensor(sample_index, dtype=torch.long),
        )


def weak_eval_augment_waveforms(clean_waveforms, max_shift=2, scale_jitter=0.02, noise_std=0.005):
    """Small random perturbations used for test-time-augmentation views."""
    waves = np.asarray(clean_waveforms, dtype=float)
    if waves.ndim == 3 and waves.shape[1] == 1:
        waves_2d = waves[:, 0, :].copy()
        add_channel_dim = True
    elif waves.ndim == 2:
        waves_2d = waves.copy()
        add_channel_dim = False
    else:
        raise ValueError(
            "clean_waveforms must have shape (n_samples, n_points) or (n_samples, 1, n_points)."
        )

    for i in range(waves_2d.shape[0]):
        if max_shift > 0:
            shift_random = np.random.randint(-max_shift, max_shift + 1)
            if shift_random > 0:
                waves_2d[i] = np.concatenate([np.zeros(shift_random), waves_2d[i, :-shift_random]])
            elif shift_random < 0:
                waves_2d[i] = np.concatenate([waves_2d[i, -shift_random:], np.zeros(-shift_random)])
        if scale_jitter > 0:
            waves_2d[i] *= np.random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
        if noise_std > 0:
            waves_2d[i] += np.random.normal(0.0, noise_std, size=waves_2d[i].shape)

    if add_channel_dim:
        return waves_2d[:, None, :]
    return waves_2d


def load_scsn_polarity_dataset(
    data_path,
    snr_range,
    n_select,
    resize=32,
    shift=0,
    bino=True,
    aug_shift=1,
    aug_noise_std_range=(0.05, 0.2),
    aug_scale_range=(0.8, 1.2),
    norm_mod="max",
    waveform_key: str = DEFAULT_WAVEFORM_KEY,
    label_key: str = DEFAULT_LABEL_KEY,
    snr_key: str = DEFAULT_SNR_KEY,
    selection_seed: Optional[int] = 42,
) -> SeismicPolarityDataset:
    """Load an SCSN polarity dataset, selecting ``n_select`` rows by SNR range.

    When ``n_select`` is positive, twice as many candidates are drawn with
    ``selection_seed`` and the first ``n_select`` rows that survive the
    binary-polarity filter are kept, so the selection is reproducible.
    """
    data_path = resolve_data_path(data_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Pass --data-path (or set data_path in the YAML config) pointing to the SCSN HDF5 file."
        )

    low, high = snr_range
    with h5py.File(data_path, "r") as handle:
        require_hdf5_keys(handle, (waveform_key, label_key, snr_key), data_path)
        snr_all = handle[snr_key][:]
    sel = np.where((snr_all >= low) & (snr_all < high))[0]

    if sel.size == 0:
        raise ValueError(f"No samples in snr range [{low}, {high}).")

    n_select = int(n_select) if n_select and n_select > 0 else None
    if n_select is not None and sel.size < n_select:
        raise ValueError(
            f"Only {sel.size} samples are in snr range [{low}, {high}), "
            f"cannot select requested n_select={n_select}."
        )

    if n_select is not None:
        rng = np.random.default_rng(selection_seed)
        candidate_count = min(sel.size, n_select * 2)
        sel = rng.choice(sel, size=candidate_count, replace=False)

    waveforms, polarities, snr_sel, _ = read_scsn_rows(
        data_path,
        sel,
        waveform_key=waveform_key,
        label_key=label_key,
        snr_key=snr_key,
    )

    if bino:
        keep_mask = np.asarray(polarities).reshape(-1) != 2
        waveforms = waveforms[keep_mask]
        polarities = polarities[keep_mask]
        snr_sel = snr_sel[keep_mask]

    if n_select is not None:
        if len(waveforms) < n_select:
            raise ValueError(
                f"Only {len(waveforms)} samples remain after filtering {len(sel)} candidates, "
                f"cannot select requested n_select={n_select}."
            )
        waveforms = waveforms[:n_select]
        polarities = polarities[:n_select]
        snr_sel = snr_sel[:n_select]

    dataset = SeismicPolarityDataset(
        waveforms=waveforms,
        polarities=polarities,
        snrs=snr_sel,
        p_arrival_times=np.full(len(waveforms), DEFAULT_P_ARRIVAL_INDEX, dtype=np.int64),
        sampling_rate=DEFAULT_SAMPLING_RATE,
        shift=shift,
        p_window=resize / DEFAULT_SAMPLING_RATE,
        apply_augmentation=True,
        aug_shift=aug_shift,
        aug_noise_std_range=aug_noise_std_range,
        aug_scale_range=aug_scale_range,
        norm_mod=norm_mod,
    )
    print(f"Data path: {data_path}")
    print(f"Dataset keys: waveform={waveform_key}, label={label_key}, snr={snr_key}")
    print("Dataset size:", len(dataset))
    return dataset


def load_ridgecrest_unlabeled_dataset(
    data_path,
    snr_range,
    n_select,
    resize=32,
    shift=0,
    aug_shift=1,
    aug_noise_std_range=(0.05, 0.2),
    aug_scale_range=(0.8, 1.2),
    norm_mod="max",
) -> SeismicPolarityDataset:
    """Load the unlabeled Ridgecrest ``phasenet`` waveforms for self-supervised
    training.

    The HDF5 file must contain a ``phasenet`` group with ``waveforms``,
    ``snr`` and ``record_id`` datasets. Rows are kept in SNR-range order (no
    label filtering is applied; polarities are set to -1). The P arrival sits
    at sample index 1000 (10 s at 100 Hz).
    """
    data_path = resolve_data_path(data_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Pass --data-path pointing to the Ridgecrest consensus HDF5 file."
        )

    low, high = snr_range
    with h5py.File(data_path, "r") as handle:
        if "phasenet" not in handle or "waveforms" not in handle["phasenet"]:
            available_keys = ", ".join(sorted(handle.keys()))
            raise KeyError(
                f"Missing 'phasenet/waveforms' group in {data_path}. "
                f"Available top-level keys: {available_keys}"
            )
        group = handle["phasenet"]
        snr_all = np.asarray(group["snr"][:]).reshape(-1)
        sel = np.where((snr_all >= float(low)) & (snr_all < float(high)))[0]
        if sel.size == 0:
            raise ValueError(f"No Ridgecrest samples in snr range [{low}, {high}).")
        sel = np.sort(sel)
        n_select = int(n_select) if n_select and n_select > 0 else None
        if n_select is not None and sel.size > n_select:
            sel = sel[:n_select]
        waveforms = np.asarray(group["waveforms"][sel], dtype=np.float32)
        snr_sel = snr_all[sel]

    dataset = SeismicPolarityDataset(
        waveforms=waveforms,
        polarities=np.full(len(waveforms), -1, dtype=np.int64),
        snrs=snr_sel,
        p_arrival_times=np.full(len(waveforms), RIDGECREST_P_ARRIVAL_INDEX, dtype=np.int64),
        sampling_rate=DEFAULT_SAMPLING_RATE,
        shift=shift,
        p_window=resize / DEFAULT_SAMPLING_RATE,
        apply_augmentation=True,
        aug_shift=aug_shift,
        aug_noise_std_range=aug_noise_std_range,
        aug_scale_range=aug_scale_range,
        norm_mod=norm_mod,
    )
    print(f"Data path: {data_path}")
    print("Dataset source: Ridgecrest unlabeled (phasenet group, no labels)")
    print("Dataset size:", len(dataset))
    return dataset


def build_train_dataloader(dataset, sample_weights=None, batch_size=256, drop_last=True):
    if sample_weights is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    weights = np.clip(weights, 0.0, None)
    if weights.shape[0] != len(dataset):
        raise ValueError("sample_weights must have the same length as dataset.")
    positive = int(np.count_nonzero(weights > 0))
    if positive == 0:
        weights = np.ones(len(dataset), dtype=np.float64)
        positive = len(dataset)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=max(positive, batch_size),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last)


def build_eval_dataloader(dataset, batch_size=256):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
