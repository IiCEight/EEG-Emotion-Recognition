"""
Diff the original (utils_PCL) data pipeline vs the refactor (data.dataloder + data.utils).

Goal: localize where the refactor's data pipeline diverges from the original.
Compares (train_data, train_labels, test_data, test_labels) for sess=0 subj=0.

Run from repo root:
    python diff_data_pipelines.py
"""
import argparse
import sys

import numpy as np
from einops import rearrange

import constant.CLI_arguments_enum as cli_enum
from data.dataloder import load_data as refactor_load_data
from data.utils import merge_and_split, normalization_wrt_subject
from main_PCL import prepare_data as orig_prepare_data


SESS = 0
SUBJ = 0


def _summary(name, a):
    print(f"  {name}: shape={a.shape} dtype={a.dtype} "
          f"sum={a.astype(np.float64).sum():.6f} "
          f"min={a.min():.6f} max={a.max():.6f} "
          f"first8={np.asarray(a).reshape(-1)[:8].tolist()}")


def _orig():
    print("[ORIG] running utils_PCL.load_data + prepare_data ...")
    args = argparse.Namespace(dataset="seed3", session=SESS, batch_size=32, device="cpu")
    target_set, source_set = orig_prepare_data(args, SUBJ)
    src = source_set["feature"].astype(np.float32)
    src_lbl = source_set["label"].astype(np.int64).reshape(-1)
    tgt = target_set["feature"].astype(np.float32)
    tgt_lbl = target_set["label"].astype(np.int64).reshape(-1)
    print(f"[ORIG] src={src.shape} tgt={tgt.shape}")
    return src, src_lbl, tgt, tgt_lbl


def _refactor():
    print("[REFACTOR] running data.dataloder.load_data + normalization_wrt_subject + merge_and_split ...")
    data, labels, num_subjects, num_electrodes, num_features, num_classes = refactor_load_data(
        dataset_name="SEED",
        dataset_path="../data/SEED",
        cache_dir="./cache",
    )
    normalization_wrt_subject(data, band_major=True)
    train_data, train_labels, test_data, test_labels = merge_and_split(
        data, labels,
        task_type=cli_enum.TaskTypeName.SUBJECT_INDEPENDENT,
        session_id=SESS,
        subject_id=SUBJ,
        split_ratio=0.6,
        data_random=False,
    )
    # (N, 62, 5) -> (N, 5, 62) -> (N, 310) band-major to match original column order
    src = rearrange(train_data, "s c f -> s f c").reshape(-1, 310).astype(np.float32)
    tgt = rearrange(test_data, "s c f -> s f c").reshape(-1, 310).astype(np.float32)
    src_lbl = np.asarray(train_labels).astype(np.int64).reshape(-1)
    tgt_lbl = np.asarray(test_labels).astype(np.int64).reshape(-1)
    print(f"[REFACTOR] src={src.shape} tgt={tgt.shape}")
    return src, src_lbl, tgt, tgt_lbl


def _compare(name, a, b):
    print(f"\n=== {name} ===")
    _summary(f"{name}.orig", a)
    _summary(f"{name}.refactor", b)
    if a.shape != b.shape:
        print(f"  !! SHAPE MISMATCH: {a.shape} vs {b.shape}")
        return
    if a.dtype != b.dtype:
        print(f"  !! DTYPE MISMATCH: {a.dtype} vs {b.dtype}")
    if np.issubdtype(a.dtype, np.integer):
        eq = np.array_equal(a, b)
        print(f"  exact_equal={eq}")
        if not eq:
            ne = np.where(a != b)[0]
            print(f"  first_diff_indices={ne[:10].tolist()} "
                  f"orig[ne[:5]]={a[ne[:5]].tolist()} ref[ne[:5]]={b[ne[:5]].tolist()}")
    else:
        diff = (a.astype(np.float64) - b.astype(np.float64))
        max_abs = np.abs(diff).max()
        mean_abs = np.abs(diff).mean()
        max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"  max_abs_diff={max_abs:.8e} mean_abs_diff={mean_abs:.8e}")
        print(f"  worst_idx={max_idx} orig={a[max_idx]:.8f} ref={b[max_idx]:.8f}")
        # row-level (per-sample) divergence: are some rows wildly different (suggesting reordering)?
        per_row = np.abs(diff).max(axis=1)
        n_close = int((per_row < 1e-5).sum())
        n_total = per_row.shape[0]
        print(f"  rows_with_max_diff<1e-5: {n_close}/{n_total}")
        # If >0 rows match exactly but others don't, suspect reordering
        if 0 < n_close < n_total:
            print(f"  !! partial match — likely sample reordering/permutation")


def main():
    o_src, o_slbl, o_tgt, o_tlbl = _orig()
    r_src, r_slbl, r_tgt, r_tlbl = _refactor()

    _compare("test_labels", o_tlbl, r_tlbl)
    _compare("test_data",   o_tgt,  r_tgt)
    _compare("train_labels", o_slbl, r_slbl)
    _compare("train_data",   o_src,  r_src)

    # If test_data matches but train doesn't, bug is in source aggregation (vstack order across subjects).
    # If both shapes match but values differ everywhere, bug is in normalization or trial reshape.
    # If shape mismatches: bug is in trial selection / merge.

    print("\n=== summary ===")
    print(f"  test_labels eq: {np.array_equal(o_tlbl, r_tlbl)}")
    print(f"  train_labels eq: {np.array_equal(o_slbl, r_slbl)}")
    print(f"  test_data shape eq: {o_tgt.shape == r_tgt.shape}")
    print(f"  train_data shape eq: {o_src.shape == r_src.shape}")
    if o_tgt.shape == r_tgt.shape:
        print(f"  test_data max_abs_diff: {np.abs(o_tgt.astype(np.float64) - r_tgt.astype(np.float64)).max():.6e}")
    if o_src.shape == r_src.shape:
        print(f"  train_data max_abs_diff: {np.abs(o_src.astype(np.float64) - r_src.astype(np.float64)).max():.6e}")


if __name__ == "__main__":
    main()
