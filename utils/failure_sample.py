"""
Utilities for tracking misclassified samples back to their original
(subject, session, trial, sample_in_trial) provenance.
"""

import csv
import os

import numpy as np


def build_test_metadata(test_data_raw: list) -> list[dict]:
    """
    Build a flat list of metadata dicts that mirrors the order produced by
    merge_for_one_subject() (i.e., ak.flatten(data, axis=1)).

    Parameters
    ----------
    test_data_raw : list
        Shape (trial, var_sample, electrode, feature) — the raw per-subject
        test data *before* merging/flattening.

    Returns
    -------
    list[dict]  one entry per merged sample, in merge order:
        {"trial": int, "sample_in_trial": int}
    """
    metadata = []
    for trial_id, trial_samples in enumerate(test_data_raw):
        trial_len = len(trial_samples)
        for sample_idx in range(trial_len):
            metadata.append({"trial": trial_id, "sample_in_trial": sample_idx, "trial_length": trial_len})
    return metadata


def record_failures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: list[dict],
    subject_id: int,
    session_id: int,
    out_path: str,
) -> None:
    """
    Append one CSV row for each misclassified sample.

    Columns: subject, session, trial, sample_in_trial, true_label, pred_label

    The file is created with a header row if it does not yet exist; otherwise
    rows are appended so that results from all LOSO folds accumulate in one file.

    Parameters
    ----------
    y_true       : ground-truth class indices, shape (N,)
    y_pred       : predicted class indices, shape (N,)
    metadata     : output of build_test_metadata(), length N
    subject_id   : held-out subject index (LOSO test subject)
    session_id   : session index
    out_path     : path to the output CSV file
    """
    write_header = not os.path.exists(out_path)

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["subject", "session", "trial", "sample_in_trial", "trial_length", "position_pct", "true_label", "pred_label"])

        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            if yt != yp:
                meta = metadata[i]
                pct = round(meta["sample_in_trial"] / meta["trial_length"] * 100, 1)
                writer.writerow([
                    subject_id,
                    session_id,
                    meta["trial"],
                    meta["sample_in_trial"],
                    meta["trial_length"],
                    pct,
                    int(yt),
                    int(yp),
                ])
