import csv
import os
import tempfile

import numpy as np
import pytest

from utils.failure_sample import build_test_metadata, record_failures


# ---------------------------------------------------------------------------
# build_test_metadata
# ---------------------------------------------------------------------------

def test_build_test_metadata_flat_order():
    # 2 trials, 3 and 2 samples respectively (electrode/feature dims don't matter)
    test_data_raw = [
        [[[0]] * 1] * 3,  # trial 0: 3 samples
        [[[0]] * 1] * 2,  # trial 1: 2 samples
    ]
    meta = build_test_metadata(test_data_raw)
    assert len(meta) == 5

    # trial indices
    assert [m["trial"] for m in meta] == [0, 0, 0, 1, 1]
    # sample_in_trial indices
    assert [m["sample_in_trial"] for m in meta] == [0, 1, 2, 0, 1]


def test_build_test_metadata_trial_length():
    test_data_raw = [
        [[[0]]] * 4,  # trial 0: 4 samples
    ]
    meta = build_test_metadata(test_data_raw)
    assert all(m["trial_length"] == 4 for m in meta)


def test_build_test_metadata_empty():
    assert build_test_metadata([]) == []


def test_build_test_metadata_single_sample():
    meta = build_test_metadata([[[[0]]]])  # 1 trial, 1 sample
    assert len(meta) == 1
    assert meta[0] == {"trial": 0, "sample_in_trial": 0, "trial_length": 1}


# ---------------------------------------------------------------------------
# record_failures
# ---------------------------------------------------------------------------

def test_record_failures_writes_header_on_new_file():
    meta = [{"trial": 0, "sample_in_trial": 0, "trial_length": 10}]
    y_true = np.array([0])
    y_pred = np.array([1])  # misclassified

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    os.remove(path)  # ensure it doesn't exist yet

    try:
        record_failures(y_true, y_pred, meta, subject_id=0, session_id=0, out_path=path)
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["subject", "session", "trial", "sample_in_trial",
                            "trial_length", "position_pct", "true_label", "pred_label"]
    finally:
        os.remove(path)


def test_record_failures_only_writes_misclassified():
    meta = [
        {"trial": 0, "sample_in_trial": 0, "trial_length": 5},
        {"trial": 0, "sample_in_trial": 1, "trial_length": 5},
        {"trial": 0, "sample_in_trial": 2, "trial_length": 5},
    ]
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 0, 2])  # only index 1 is wrong

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    os.remove(path)

    try:
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


def test_record_failures_no_failures_writes_only_header():
    meta = [{"trial": 0, "sample_in_trial": 0, "trial_length": 3}]
    y_true = np.array([1])
    y_pred = np.array([1])  # correct

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    os.remove(path)

    try:
        record_failures(y_true, y_pred, meta, subject_id=0, session_id=0, out_path=path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # only header
    finally:
        os.remove(path)


def test_record_failures_appends_without_duplicate_header():
    meta = [{"trial": 0, "sample_in_trial": 0, "trial_length": 2}]
    y_true = np.array([0])
    y_pred = np.array([1])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    os.remove(path)

    try:
        # Write twice (simulating two LOSO folds)
        record_failures(y_true, y_pred, meta, subject_id=0, session_id=0, out_path=path)
        record_failures(y_true, y_pred, meta, subject_id=1, session_id=0, out_path=path)
        with open(path) as f:
            rows = list(csv.reader(f))
        # 1 header + 2 data rows
        assert len(rows) == 3
        assert rows[0][0] == "subject"  # header only once
    finally:
        os.remove(path)


def test_record_failures_position_pct():
    # sample_in_trial=2 out of trial_length=4 → 50.0%
    meta = [{"trial": 0, "sample_in_trial": 2, "trial_length": 4}]
    y_true = np.array([0])
    y_pred = np.array([1])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    os.remove(path)

    try:
        record_failures(y_true, y_pred, meta, subject_id=0, session_id=0, out_path=path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert rows[1][5] == "50.0"  # position_pct column
    finally:
        os.remove(path)
