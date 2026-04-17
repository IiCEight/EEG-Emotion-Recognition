import csv
import os
import tempfile

import numpy as np

from utils.failure_sample import build_test_metadata, init_failure_log, record_failures


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tmp_path():
    """Return a temp file path that does not yet exist."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    os.remove(path)
    return path


# ---------------------------------------------------------------------------
# build_test_metadata
# ---------------------------------------------------------------------------

def test_build_test_metadata_flat_order():
    test_data_raw = [
        [[[0]] * 1] * 3,  # trial 0: 3 samples
        [[[0]] * 1] * 2,  # trial 1: 2 samples
    ]
    meta = build_test_metadata(test_data_raw)
    assert len(meta) == 5
    assert [m["trial"] for m in meta] == [0, 0, 0, 1, 1]
    assert [m["sample_in_trial"] for m in meta] == [0, 1, 2, 0, 1]


def test_build_test_metadata_trial_length():
    meta = build_test_metadata([[[[0]]] * 4])
    assert all(m["trial_length"] == 4 for m in meta)


def test_build_test_metadata_empty():
    assert build_test_metadata([]) == []


def test_build_test_metadata_single_sample():
    meta = build_test_metadata([[[[0]]]])
    assert len(meta) == 1
    assert meta[0] == {"trial": 0, "sample_in_trial": 0, "trial_length": 1}


# ---------------------------------------------------------------------------
# init_failure_log
# ---------------------------------------------------------------------------

def test_init_failure_log_creates_file_with_header():
    path = _tmp_path()
    try:
        init_failure_log(path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1
        assert rows[0] == ["subject", "session", "trial", "sample_in_trial",
                            "trial_length", "position_pct", "true_label", "pred_label"]
    finally:
        os.remove(path)


def test_init_failure_log_overwrites_existing_file():
    path = _tmp_path()
    try:
        # Write some content first
        with open(path, "w") as f:
            f.write("stale,data\nrow1\nrow2\n")
        init_failure_log(path)
        with open(path) as f:
            rows = list(csv.reader(f))
        # Only the header remains
        assert len(rows) == 1
        assert rows[0][0] == "subject"
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# record_failures
# ---------------------------------------------------------------------------

def test_record_failures_only_writes_misclassified():
    meta = [
        {"trial": 0, "sample_in_trial": 0, "trial_length": 5},
        {"trial": 0, "sample_in_trial": 1, "trial_length": 5},
        {"trial": 0, "sample_in_trial": 2, "trial_length": 5},
    ]
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 0, 2])  # only index 1 is wrong

    path = _tmp_path()
    try:
        init_failure_log(path)
        record_failures(y_true, y_pred, meta, subject_id=1, session_id=2, out_path=path)
        with open(path) as f:
            rows = list(csv.reader(f))
        # header + 1 failure row
        assert len(rows) == 2
        row = rows[1]
        assert row[0] == "1"   # subject
        assert row[1] == "2"   # session
        assert row[2] == "0"   # trial
        assert row[3] == "1"   # sample_in_trial
        assert row[6] == "1"   # true_label
        assert row[7] == "0"   # pred_label
    finally:
        os.remove(path)


def test_record_failures_no_failures_appends_nothing():
    meta = [{"trial": 0, "sample_in_trial": 0, "trial_length": 3}]
    y_true = np.array([1])
    y_pred = np.array([1])  # correct

    path = _tmp_path()
    try:
        init_failure_log(path)
        record_failures(y_true, y_pred, meta, subject_id=0, session_id=0, out_path=path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # only header, no data rows
    finally:
        os.remove(path)


def test_record_failures_appends_across_loso_folds():
    meta = [{"trial": 0, "sample_in_trial": 0, "trial_length": 2}]
    y_true = np.array([0])
    y_pred = np.array([1])

    path = _tmp_path()
    try:
        init_failure_log(path)
        # Two LOSO folds
        record_failures(y_true, y_pred, meta, subject_id=0, session_id=0, out_path=path)
        record_failures(y_true, y_pred, meta, subject_id=1, session_id=0, out_path=path)
        with open(path) as f:
            rows = list(csv.reader(f))
        # 1 header + 2 data rows, no duplicate header
        assert len(rows) == 3
        assert rows[0][0] == "subject"
        assert rows[1][0] == "0"
        assert rows[2][0] == "1"
    finally:
        os.remove(path)


def test_record_failures_position_pct():
    # sample_in_trial=2 out of trial_length=4 → 50.0%
    meta = [{"trial": 0, "sample_in_trial": 2, "trial_length": 4}]
    y_true = np.array([0])
    y_pred = np.array([1])

    path = _tmp_path()
    try:
        init_failure_log(path)
        record_failures(y_true, y_pred, meta, subject_id=0, session_id=0, out_path=path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert rows[1][5] == "50.0"
    finally:
        os.remove(path)
