import numpy as np
import pytest
from data.load.load_deap import _compute_de


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
