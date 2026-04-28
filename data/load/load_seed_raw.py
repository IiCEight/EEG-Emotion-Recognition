"""
Load raw SEED EEG from Preprocessed_EEG/, compute DE per frequency band
over configurable windows, and apply LDS smoothing.

Output shape matches load_seed():
    data:  (session=3, subject=15, trial=15, sample=var, electrode=62, band=5)
    label: (session=3, subject=15, trial=15, sample=var)
"""

import multiprocessing as mp
from functools import partial

import numpy as np
from loguru import logger
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

# --- constants ---------------------------------------------------------------

SAMPLING_RATE = 200  # Hz (SEED Preprocessed_EEG is downsampled to 200 Hz)

# 5 frequency bands used in SEED (matches ExtractedFeatures ordering)
_BANDS = [
    (1,  4),   # delta
    (4,  8),   # theta
    (8,  14),  # alpha
    (14, 31),  # beta
    (31, 50),  # gamma
]

_FILTER_ORDER = 4
_LDS_ALPHA = 0.5  # exponential smoothing weight (SEED convention)

# Same session/subject file mapping as load_seed.py
_EEG_FILES = [
    ['1_20131027.mat',  '2_20140404.mat',  '3_20140603.mat',
     '4_20140621.mat',  '5_20140411.mat',  '6_20130712.mat',
     '7_20131027.mat',  '8_20140511.mat',  '9_20140620.mat',
     '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
     '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],
    ['1_20131030.mat',  '2_20140413.mat',  '3_20140611.mat',
     '4_20140702.mat',  '5_20140418.mat',  '6_20131016.mat',
     '7_20131030.mat',  '8_20140514.mat',  '9_20140627.mat',
     '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
     '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],
    ['1_20131107.mat',  '2_20140419.mat',  '3_20140629.mat',
     '4_20140705.mat',  '5_20140506.mat',  '6_20131113.mat',
     '7_20131106.mat',  '8_20140521.mat',  '9_20140704.mat',
     '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
     '13_20140610.mat', '14_20140627.mat', '15_20131105.mat'],
]

# --- internal helpers --------------------------------------------------------

def _bandpass_sos(low: float, high: float, fs: int, order: int = _FILTER_ORDER):
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype='band', output='sos')


def _compute_de(signal: np.ndarray) -> float:
    """DE of a 1-D signal: 0.5 * log(2πe * var)."""
    return 0.5 * np.log(2 * np.pi * np.e * np.var(signal) + 1e-10)


def _apply_lds(de_seq: np.ndarray, alpha: float = _LDS_ALPHA) -> np.ndarray:
    """
    Exponential smoothing along time axis (first axis).
    de_seq shape: (T, electrode, band)
    """
    smoothed = np.empty_like(de_seq)
    smoothed[0] = de_seq[0]
    for t in range(1, len(de_seq)):
        smoothed[t] = alpha * de_seq[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def _process_trial(
    raw: np.ndarray,          # (62, T_raw)
    window_samples: int,
    stride_samples: int,
    sos_filters: list,
) -> np.ndarray:
    """
    For one trial: bandpass → segment → DE → LDS.

    Returns shape (n_windows, 62, 5).
    """
    n_electrodes, T = raw.shape
    n_windows = max(0, (T - window_samples) // stride_samples + 1)
    if n_windows == 0:
        return np.empty((0, n_electrodes, len(_BANDS)), dtype=np.float32)

    de_windows = np.zeros((n_windows, n_electrodes, len(_BANDS)), dtype=np.float32)

    for band_idx, sos in enumerate(sos_filters):
        filtered = sosfiltfilt(sos, raw, axis=1)  # (62, T)
        for w in range(n_windows):
            start = w * stride_samples
            seg = filtered[:, start: start + window_samples]  # (62, window_samples)
            for e in range(n_electrodes):
                de_windows[w, e, band_idx] = _compute_de(seg[e])

    return _apply_lds(de_windows)  # (n_windows, 62, 5)


def _read_subject(dir_path: str, window_samples: int, stride_samples: int, file: str) -> list:
    """Read one subject file, return list of 15 trials each (n_windows, 62, 5)."""
    subject_data = loadmat(f"{dir_path}/{file}")
    keys = list(subject_data.keys())[3:]  # skip __header__, __version__, __globals__

    sos_filters = [_bandpass_sos(lo, hi, SAMPLING_RATE) for lo, hi in _BANDS]

    trials = []
    for i in range(15):
        raw = subject_data[keys[i]][:, 1:]  # (62, T_raw) — drop dummy first column
        trial_de = _process_trial(raw, window_samples, stride_samples, sos_filters)
        trials.append(trial_de.tolist())
    return trials


# --- public API --------------------------------------------------------------

def load_seed_raw(
    dataset_path: str,
    sample_length: int = 1,    # window size in seconds
    stride: int | None = None, # step in seconds; defaults to sample_length (non-overlapping)
) -> tuple[list, list, int, int, int, int]:
    """
    Load SEED from Preprocessed_EEG/, compute windowed DE+LDS features.

    Returns the same signature as load_seed():
        data:  list (session=3, subject=15, trial=15, sample=var, electrode=62, band=5)
        label: list (session=3, subject=15, trial=15, sample=var)
        num_subjects=15, num_electrodes=62, num_bands=5, num_classes=3
    """
    _stride = stride if stride is not None else sample_length
    window_samples = sample_length * SAMPLING_RATE
    stride_samples = _stride * SAMPLING_RATE

    logger.info(
        "load_seed_raw: sample_length={}s  stride={}s  window_samples={}  stride_samples={}",
        sample_length, _stride, window_samples, stride_samples,
    )

    dir_path = dataset_path.rstrip("/") + "/Preprocessed_EEG"

    # Labels: shape (1, 15) with values -1/0/1 → shift to 0/1/2, replicate to (3, 15, 15)
    raw_label = np.array(loadmat(f"{dir_path}/label.mat")["label"])
    labels = np.tile(raw_label[0] + 1, (3, 15, 1)).tolist()  # (3, 15, 15)
    num_classes = 3

    eeg_data = [None] * 3
    for session_id, session_files in enumerate(_EEG_FILES):
        with mp.Pool(processes=5) as pool:
            results = pool.map(
                partial(_read_subject, dir_path, window_samples, stride_samples),
                session_files,
            )
        eeg_data[session_id] = results  # list of 15 subjects, each list of 15 trials

    # Expand labels: one label per window (all windows in a trial share the trial label)
    for session_id in range(3):
        for subject_id in range(15):
            for trial_id in range(15):
                trial_label = labels[session_id][subject_id][trial_id]
                n_windows = len(eeg_data[session_id][subject_id][trial_id])
                labels[session_id][subject_id][trial_id] = [trial_label] * n_windows

    return eeg_data, labels, 15, 62, 5, num_classes
