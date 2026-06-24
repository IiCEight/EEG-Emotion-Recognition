import numpy as np
import pytest
from data.load.load_deap import _compute_de, _segment, _subtract_baseline


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
