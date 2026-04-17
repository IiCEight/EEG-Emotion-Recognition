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


_CSV_HEADER = ["subject", "session", "trial", "sample_in_trial",
               "trial_length", "position_pct", "true_label", "pred_label"]


def init_failure_log(out_path: str) -> None:
    """
    Create (or overwrite) the failure CSV with just the header row.

    Call once before the LOSO loop so that each experiment run starts clean.

    Parameters
    ----------
    out_path : path to the output CSV file
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerow(_CSV_HEADER)


def record_failures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: list[dict],
    subject_id: int,
    session_id: int,
    out_path: str,
) -> None:
    """
    Append one CSV row per misclassified sample for one subject/session fold.

    Call this ONCE per fold (after training ends) with the predictions from
    the best epoch, not on every new best — otherwise stale rows accumulate.

    The file must already exist (created by init_failure_log); rows are
    appended so results from all LOSO folds accumulate in one file.

    Parameters
    ----------
    y_true       : ground-truth class indices, shape (N,)
    y_pred       : predicted class indices, shape (N,)
    metadata     : output of build_test_metadata(), length N
    subject_id   : held-out subject index (LOSO test subject)
    session_id   : session index
    out_path     : path to the output CSV file
    """
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
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
