import pickle
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from loguru import logger

NUM_SUBJECTS = 32
NUM_TRIALS = 40
SAMPLE_RATE = 128
NUM_ELECTRODES = 32
NUM_BANDS = 5
BASELINE_DURATION = 3   # seconds
STIMULUS_DURATION = 60  # seconds

_BANDS = [(1, 4), (4, 8), (8, 13), (13, 31), (31, 50)]

_LABEL_IDX = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}


def _bandpass(data: np.ndarray, low: float, high: float) -> np.ndarray:
    """Zero-phase order-3 Butterworth bandpass. data: (..., samples)."""
    nyq = SAMPLE_RATE / 2.0
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def _compute_de(segments: np.ndarray) -> np.ndarray:
    """
    Differential Entropy per window per channel per band.

    segments: (N, C, W)
    returns:  (N, C, 5)
    """
    n, c, _ = segments.shape
    features = np.empty((n, c, NUM_BANDS), dtype=np.float64)
    for b_i, (low, high) in enumerate(_BANDS):
        filtered = _bandpass(segments, low, high)          # (N, C, W)
        variance = np.var(filtered, ddof=1, axis=-1)       # (N, C)
        features[:, :, b_i] = 0.5 * np.log(2 * np.pi * np.e * variance)
    return features


def _segment(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment (T, C, S) into non-overlapping 1-second windows.

    Returns:
        groups:   (T * n_slices,)  int — 1-indexed trial IDs
        segments: (T * n_slices, C, 128)
    """
    window = SAMPLE_RATE  # 128 samples
    T, C, S = signal.shape
    n_slices = (S - window) // window + 1
    # stack → (T, n_slices, C, window) then flatten trials
    slices = np.stack(
        [signal[:, :, i * window: i * window + window] for i in range(n_slices)],
        axis=1,
    )  # (T, n_slices, C, window)
    segments = slices.reshape(T * n_slices, C, window)
    groups = np.repeat(np.arange(1, T + 1), n_slices)
    return groups, segments


def _subtract_baseline(
    stimulus_de: np.ndarray,
    stim_groups: np.ndarray,
    baseline_de: np.ndarray,
    base_groups: np.ndarray,
) -> np.ndarray:
    """Per-trial baseline correction. All arrays: (N, C, F) / (N,)."""
    corrected = np.zeros_like(stimulus_de)
    for trial in np.unique(base_groups):
        base_mean = baseline_de[base_groups == trial].mean(axis=0)  # (C, F)
        mask = stim_groups == trial
        corrected[mask] = stimulus_de[mask] - base_mean
    return corrected


def _lds(data: np.ndarray) -> np.ndarray:
    """
    Kalman-filter (LDS) smoothing per trial.

    data: (T, C, F)  — time windows × channels × features
    returns same shape.
    Ported from TransEER / LibEER.
    """
    num_t, num_channel, num_feature = data.shape
    x = data.reshape(num_t, -1).T   # (C*F, T)

    prior_correlation = 0.01
    noise_correlation = 0.0001
    observation_correlation = 1

    mean = x.mean(axis=1)           # (C*F,)
    num_features, num_samples = x.shape

    P = np.zeros_like(x)
    U = np.zeros_like(x)
    K = np.zeros_like(x)
    V = np.zeros_like(x)

    K[:, 0] = prior_correlation / (prior_correlation + observation_correlation)
    U[:, 0] = mean + K[:, 0] * (x[:, 0] - prior_correlation)
    V[:, 0] = (1 - K[:, 0]) * prior_correlation

    for i in range(1, num_samples):
        P[:, i - 1] = V[:, i - 1] + noise_correlation
        K[:, i] = P[:, i - 1] / (P[:, i - 1] + observation_correlation)
        U[:, i] = U[:, i - 1] + K[:, i] * (x[:, i] - U[:, i - 1])
        V[:, i] = (1 - K[:, i]) * P[:, i - 1]

    return U.T.reshape(num_t, num_channel, num_feature)

