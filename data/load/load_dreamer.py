import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from loguru import logger

# DREAMER: 23 subjects, 18 trials, 1 session, 128 Hz, 14 electrodes
NUM_SUBJECTS = 23
NUM_TRIALS = 18
SAMPLE_RATE = 128
NUM_ELECTRODES = 14
NUM_BANDS = 5

# Band definitions: (low_hz, high_hz) matching SEED's 5-band DE
BANDS = [
    (1, 4),    # delta
    (4, 8),    # theta
    (8, 14),   # alpha
    (14, 31),  # beta
    (31, 50),  # gamma
]

# DREAMER label scores are 1–5; map to 3 classes: 0=low(1-2), 1=neutral(3), 2=high(4-5)
def _score_to_class(score: int) -> int:
    if score <= 2:
        return 0
    elif score == 3:
        return 1
    else:
        return 2


def _bandpass(data: np.ndarray, low: float, high: float, fs: int = 128) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass to (channels, samples)."""
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)


def _de_per_window(segment: np.ndarray) -> np.ndarray:
    """
    Compute Differential Entropy per electrode per band for one 1-second window.
    segment shape: (electrodes, samples_per_second)
    returns: (electrodes, num_bands)
    """
    result = np.zeros((segment.shape[0], NUM_BANDS), dtype=np.float32)
    for b_i, (low, high) in enumerate(BANDS):
        filtered = _bandpass(segment, low, high, SAMPLE_RATE)
        var = np.var(filtered, axis=1) + 1e-8
        result[:, b_i] = 0.5 * np.log(2 * np.pi * np.e * var)
    return result


def _extract_de_windows(raw: np.ndarray) -> list:
    """
    raw shape: (electrodes, total_samples)
    Returns list of (electrodes, num_bands) arrays, one per 1-second window.
    """
    n_samples = raw.shape[1]
    n_windows = n_samples // SAMPLE_RATE
    windows = []
    for w in range(n_windows):
        seg = raw[:, w * SAMPLE_RATE:(w + 1) * SAMPLE_RATE]
        windows.append(_de_per_window(seg))
    return windows  # list of (E, F)


def load_dreamer(
    dir_path: str,
    label_type: str = "valence",
    trim_trial_start_pct: float = 0.0,
) -> tuple[list, list, int, int, int, int]:
    """
    Load and preprocess the DREAMER dataset.

    Returns the same contract as load_seed:
        data:  list, shape (session=1, subject=23, trial=18, sample, electrode=14, band=5)
        label: list, shape (session=1, subject=23, trial=18, sample)  — integer 0/1/2
        num_subjects, num_electrodes, num_features, num_classes
    """
    label_idx = {"valence": 0, "arousal": 1, "dominance": 2}
    if label_type not in label_idx:
        raise ValueError(f"label_type must be one of {list(label_idx)}, got '{label_type}'")
    l_idx = label_idx[label_type]

    file_path = dir_path + "/../data/DREAMER.mat"
    logger.info("Loading DREAMER dataset from {}", file_path)
    mat = loadmat(file_path)["DREAMER"]

    # One session wrapping the full dataset
    data = [[[] for _ in range(NUM_SUBJECTS)]]
    labels = [[[] for _ in range(NUM_SUBJECTS)]]

    for sub in range(NUM_SUBJECTS):
        sub_data_obj = mat[0, 0]["Data"][0, sub]
        stimuli_obj = sub_data_obj["EEG"][0, 0]["stimuli"][0, 0]
        baseline_obj = sub_data_obj["EEG"][0, 0]["baseline"][0, 0]
        scores = np.array([
            sub_data_obj["ScoreValence"][0, 0].flatten(),
            sub_data_obj["ScoreArousal"][0, 0].flatten(),
            sub_data_obj["ScoreDominance"][0, 0].flatten(),
        ])  # (3, 18)

        sub_trials_data = []
        sub_trials_labels = []

        for trial in range(NUM_TRIALS):
            # raw shape from mat: (total_samples, electrodes) → transpose to (electrodes, samples)
            stim_raw = stimuli_obj[trial, 0].T.astype(np.float64)   # (14, S_stim)
            base_raw = baseline_obj[trial, 0].T.astype(np.float64)  # (14, S_base)

            # DE features: list of (14, 5) windows
            stim_windows = _extract_de_windows(stim_raw)
            base_windows = _extract_de_windows(base_raw)

            if len(stim_windows) == 0:
                logger.warning("Subject {} trial {} has no stimulus windows, skipping", sub, trial)
                continue

            stim_arr = np.stack(stim_windows, axis=0)  # (T, 14, 5)

            # Baseline subtraction: subtract mean DE across all baseline windows
            if len(base_windows) > 0:
                base_mean = np.mean(np.stack(base_windows, axis=0), axis=0)  # (14, 5)
                stim_arr = stim_arr - base_mean[np.newaxis]

            if trim_trial_start_pct > 0.0:
                skip = int(len(stim_arr) * trim_trial_start_pct / 100.0)
                stim_arr = stim_arr[skip:]

            cls = _score_to_class(int(scores[l_idx, trial]))
            trial_labels = [cls] * len(stim_arr)

            sub_trials_data.append(stim_arr.tolist())
            sub_trials_labels.append(trial_labels)

        data[0][sub] = sub_trials_data
        labels[0][sub] = sub_trials_labels

    num_classes = 3
    logger.info(
        "DREAMER loaded: {} subjects, {} trials/subject, {} classes ({})",
        NUM_SUBJECTS, NUM_TRIALS, num_classes, label_type,
    )
    return data, labels, NUM_SUBJECTS, NUM_ELECTRODES, NUM_BANDS, num_classes
