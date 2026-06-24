import numpy as np
import pytest
from data.load.load_deap import _compute_de, _segment, _subtract_baseline, _lds


def test_compute_de_output_shape():
    rng = np.random.default_rng(0)
    # 10 windows, 32 channels, 128 samples (1 s @ 128 Hz)
    segments = rng.standard_normal((10, 32, 128))
    result = _compute_de(segments)
    assert result.shape == (10, 32, 5)


def test_compute_de_output_dtype():
    rng = np.random.default_rng(1)
    segments = rng.standard_normal((4, 3, 128))
    result = _compute_de(segments)
    assert result.dtype in (np.float64, np.float32)


def test_compute_de_finite():
    rng = np.random.default_rng(2)
    segments = rng.standard_normal((8, 5, 128))
    result = _compute_de(segments)
    assert np.all(np.isfinite(result))


def test_segment_shape_stimulus():
    # 40 trials, 32 channels, 7680 samples (60 s @ 128 Hz) → 60 windows/trial
    signal = np.zeros((40, 32, 7680))
    groups, segs = _segment(signal)
    assert segs.shape == (40 * 60, 32, 128)
    assert groups.shape == (40 * 60,)


def test_segment_shape_baseline():
    # 40 trials, 32 channels, 384 samples (3 s @ 128 Hz) → 3 windows/trial
    signal = np.zeros((40, 32, 384))
    groups, segs = _segment(signal)
    assert segs.shape == (40 * 3, 32, 128)
    assert groups.shape == (40 * 3,)


def test_segment_groups_are_1indexed():
    signal = np.zeros((3, 2, 256))   # 3 trials, 2 windows each
    groups, _ = _segment(signal)
    assert list(groups) == [1, 1, 2, 2, 3, 3]


def test_segment_window_content():
    # Each trial filled with its trial index so we can verify slicing is correct
    signal = np.zeros((3, 1, 256))
    for t in range(3):
        signal[t, 0, :] = t
    _, segs = _segment(signal)
    # window 0 and 1 should both contain trial-0 values
    assert segs[0, 0, 0] == 0.0
    assert segs[1, 0, 0] == 0.0
    assert segs[2, 0, 0] == 1.0


def test_subtract_baseline_zeros_out_constant():
    # baseline mean = 2.0, stimulus = 2.0 → corrected should be 0.0
    base_de = np.full((6, 4, 5), 2.0)   # 2 trials × 3 windows each
    base_groups = np.array([1, 1, 1, 2, 2, 2])
    stim_de = np.full((8, 4, 5), 2.0)   # 2 trials × 4 windows each
    stim_groups = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    result = _subtract_baseline(stim_de, stim_groups, base_de, base_groups)
    assert result.shape == stim_de.shape
    np.testing.assert_allclose(result, 0.0)


def test_subtract_baseline_per_trial_independence():
    # trial 1 baseline mean = 1.0, trial 2 baseline mean = 3.0
    base_de = np.concatenate([
        np.full((3, 2, 5), 1.0),
        np.full((3, 2, 5), 3.0),
    ], axis=0)
    base_groups = np.array([1, 1, 1, 2, 2, 2])
    stim_de = np.concatenate([
        np.full((4, 2, 5), 5.0),
        np.full((4, 2, 5), 5.0),
    ], axis=0)
    stim_groups = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    result = _subtract_baseline(stim_de, stim_groups, base_de, base_groups)
    np.testing.assert_allclose(result[:4], 4.0)   # 5 - 1
    np.testing.assert_allclose(result[4:], 2.0)   # 5 - 3


def test_lds_shape_preserved():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((60, 32, 5))
    result = _lds(x)
    assert result.shape == (60, 32, 5)


def test_lds_output_finite():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((60, 32, 5))
    result = _lds(x)
    assert np.all(np.isfinite(result))


def test_lds_smoothing_effect():
    # Constant signal should be unchanged by the Kalman smoother
    x = np.ones((20, 4, 5)) * 3.0
    result = _lds(x)
    # After convergence the smoother should reproduce ~constant output
    np.testing.assert_allclose(result[5:], 3.0, atol=0.05)


import os, tempfile, pickle
from data.load.load_deap import load_deap


def _make_fake_subject(path: str, subject_id: int):
    """Write a minimal synthetic .dat file matching DEAP pickle format."""
    rng = np.random.default_rng(subject_id)
    # (40 trials, 40 channels, 8064 samples)
    data = rng.standard_normal((40, 40, 8064)).astype(np.float64)
    labels = rng.uniform(1, 9, (40, 4)).astype(np.float64)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"s{subject_id:02d}.dat")
    with open(file_path, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    return file_path


def test_load_deap_shape_two_subjects():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_subject(tmp, 1)
        _make_fake_subject(tmp, 2)
        data, labels, n_subj, n_elec, n_feat, n_cls = load_deap(
            tmp, label_type="valence"
        )
    # session dim = 1
    assert len(data) == 1
    assert len(labels) == 1
    # 2 subjects loaded
    assert len(data[0]) == 2
    assert len(labels[0]) == 2
    # 40 trials per subject
    assert len(data[0][0]) == 40
    # each trial has 60 samples (60 s × 1 window/s)
    assert len(data[0][0][0]) == 60
    # each sample: 32 electrodes × 5 bands
    assert np.array(data[0][0][0][0]).shape == (32, 5)
    # metadata
    assert n_subj == 2
    assert n_elec == 32
    assert n_feat == 5
    assert n_cls == 2


def test_load_deap_labels_binary():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_subject(tmp, 1)
        _, labels, *_ = load_deap(tmp, label_type="valence")
    for trial_labels in labels[0][0]:
        for lbl in trial_labels:
            assert lbl in (0, 1)


def test_load_deap_arousal_label():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_subject(tmp, 1)
        _, labels_v, *_ = load_deap(tmp, label_type="valence")
        _, labels_a, *_ = load_deap(tmp, label_type="arousal")
    # valence and arousal can differ (not guaranteed equal)
    v = labels_v[0][0][0][0]
    a = labels_a[0][0][0][0]
    assert v in (0, 1) and a in (0, 1)


def test_load_deap_trim():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_subject(tmp, 1)
        data_full, _, *_ = load_deap(tmp, label_type="valence", trim_trial_start_pct=0.0)
        data_trim, _, *_ = load_deap(tmp, label_type="valence", trim_trial_start_pct=50.0)
    full_len = len(data_full[0][0][0])   # 60
    trim_len = len(data_trim[0][0][0])   # 30
    assert trim_len == full_len // 2

