

import awkward as ak
from einops import rearrange
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from constant import CLI_arguments_enum


def merge_and_split(data:list, labels:list, task_type, session_id, subject_id, split_ratio, data_random,
                    time_steps: int = 1,
                    )-> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if task_type == CLI_arguments_enum.TaskTypeName.SUBJECT_DEPENDENT:
        train_data, train_labels, test_data, test_labels = (
            split_data_wrt_trials(
                data[session_id][subject_id],
                labels[session_id][subject_id], split_ratio, data_random)
        )
        # merge train data and labels
        train_data, train_labels = merge_for_one_subject(train_data, train_labels, time_steps)
        # We keep session dimension for test data,
        # since we want to test on all sessions separately.
        test_data, test_labels = merge_for_one_subject(test_data, test_labels, time_steps)

    else:
        # For subject-independent setting, we leave current subject out as test data
        # and merge the rest subjects' data as train data.
        train_data, train_labels, test_data, test_labels = (
            split_data_wrt_subjects(
                data[session_id], labels[session_id], subject_id)
        )

        train_data, train_labels = merge_for_all_subjects(train_data, train_labels, time_steps=time_steps)

        test_data, test_labels = merge_for_one_subject(test_data, test_labels, time_steps)

    return train_data, train_labels, test_data, test_labels


def _group_into_windows(trial_data: np.ndarray, time_steps: int) -> np.ndarray:
    """Group (N, ...) → (N//T, T, ...), dropping remainder. No-op when time_steps=1."""
    if time_steps == 1:
        return trial_data
    n_win = len(trial_data) // time_steps
    arr = ak.Array(trial_data[:n_win * time_steps])
    return np.array(ak.unflatten(arr, time_steps, axis=0))  # (n_win, T, ...)


def merge_for_all_subjects(
    data: list,
    labels: list,
    merge_subject_dim = True,
    time_steps: int = 1,
)-> tuple[np.ndarray, np.ndarray]:
    """
    input:
        data:  list, shape (subject, trial, sample, electrode, feature)
        label: list, shape (subject, trial, sample)

    return:
        if not merge_subject_dim:
            data:  np.ndarray, shape (subject, new_sample (trial * sample), electrode, feature)
            label: np.ndarray, shape (subject, new_sample (trial * sample))

        else:
            data:  np.ndarray, shape (new_sample (subject  * trial * sample), electrode, feature)
            label: np.ndarray, shape (new_sample (subject * trial * sample))
    """
    logger.info(f"Merging data and labels....")

    grouped, grouped_labels = [], []
    for sub_trials, sub_labels in zip(data, labels):
        sub_grouped, sub_gl = [], []
        for trial, tlabels in zip(sub_trials, sub_labels):
            t = np.array(trial)       # (N, E, F)
            l = np.array(tlabels)     # (N,)
            t = _group_into_windows(t, time_steps)      # (n_win, T, E, F) or (N, E, F)
            n_win = len(t)
            sub_grouped.append(t)
            sub_gl.append(l[:n_win * time_steps:time_steps])
        grouped.append(np.concatenate(sub_grouped, axis=0))
        grouped_labels.append(np.concatenate(sub_gl, axis=0))
    data = np.stack(grouped, axis=0)
    labels = np.stack(grouped_labels, axis=0)

    if not merge_subject_dim:
        logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)
        return data, labels

    if merge_subject_dim:
        data = rearrange(data, "subject samples ... -> (subject samples) ...")
        labels = rearrange(labels, "subject samples -> (subject samples)")

    logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)

    return data, labels

def merge_for_one_subject(
    data: list,
    labels: list,
    time_steps: int = 1,
)-> tuple[np.ndarray, np.ndarray]:
    """
    input:
        data: list, shape (trial, sample(may different), electrode, feature)
        label: list, shape (trial, sample)

    return:
        data: np.ndarray, shape (new_sample (trial * sample), electrode, feature)
        label: np.ndarray, shape (new_sample (trial * sample))
    """
    logger.info(f"Merging data and labels....")

    grouped, grouped_labels = [], []
    for trial, tlabels in zip(data, labels):
        t = np.array(trial)       # (N, E, F)
        l = np.array(tlabels)     # (N,)
        t = _group_into_windows(t, time_steps)      # (n_win, T, E, F) or (N, E, F)
        n_win = len(t)
        grouped.append(t)
        grouped_labels.append(l[:n_win * time_steps:time_steps])
    data = np.concatenate(grouped, axis=0)
    labels = np.concatenate(grouped_labels, axis=0)

    logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)

    return data, labels
    
def split_data_wrt_trials(data:list, labels:list, split_ratio:float, random = False
                          )->tuple[list, list, list, list]:
    """
    For one subject, we split 60% of the trials as train data 
        and the rest 40% as test data.
        
    input:
        data: list, shape (trial, sample, electrode, feature)
        label: list, shape (trial, sample)
    return:
        train_data: list, the same as input
        train_label: list, the same as input
        test_data: list, the same as input
        test_label: list, the same as input
    """
    data = ak.Array(data)
    labels = ak.Array(labels)

    # number of trails for each subject,
    num_trials = len(data)
    logger.debug(f"Number of trials: {num_trials}")
    num_trails_per_train = int(num_trials * split_ratio)
    logger.info("Number of trials for train {}", num_trails_per_train)
    mask = [True] * num_trails_per_train + [False] * (num_trials - num_trails_per_train)

    mask = np.array(mask)

    if random:
        np.random.shuffle(mask)

    # Use bool index of ak
    # convert back to list
    train_data = data[mask].to_list()
    train_labels = labels[mask].to_list()
    test_data = data[~mask].to_list()
    test_labels = labels[~mask].to_list()

    
    return train_data, train_labels, test_data, test_labels


def split_data_wrt_subjects(data:list, labels:list, subject_id:int
                          )->tuple[list, list, list, list]:
    """
    leave one subject(subject_id:int) as test set.
            
    input:
        data:  list, shape (subject, trial, sample, electrode, feature)
        label: list, shape (subject, trial, sample)
    return:
        train_data:  list, shape (subject - 1, trial, sample, electrode, feature)
        train_label: list, shape (subject - 1, trial, sample)
        test_data:   list, shape (trial, sample, electrode, feature)
        test_label:  list, shape (trial, sample)
    """
    # We need the function of boolean array index.
    data = ak.Array(data)
    labels = ak.Array(labels)

    # number for each subject,
    num_subjects = len(labels)

    mask = np.ones(num_subjects, dtype=bool)
    mask[subject_id] = False

    # convert back to list
    train_data = data[mask].to_list()
    train_labels = labels[mask].to_list()
    test_data = data[subject_id].to_list()
    test_labels = labels[subject_id].to_list()
    
    return train_data, train_labels, test_data, test_labels


def normalization_wrt_trial(data:list, type = 'min_max'):
    '''
    param {type}: min_max, z_score
    
    input:
        data: list shape (session, subject, trial, sample, electrode, feature)
        type: str, 'min_max' or 'z_score'

    return:
        data the same shape as input
    '''

    for session_id in range(len(data)):
        for subject_id in range(len(data[session_id])):
            for trial_id in range(len(data[session_id][subject_id])):
                    data[session_id][subject_id][trial_id]= normalize_one_trial(
                        data[session_id][subject_id][trial_id], type)

    return data

def normalization_wrt_session(data:list, type = 'min_max'):
    '''
    param {type}: min_max, z_score
    
    input:
        data: list shape (session, subject, trial, sample, electrode, feature)
        type: str, 'min_max' or 'z_score'

    return:
        data the same shape as input
    '''

    for session_id in range(len(data)):
        for subject_id in range(len(data[session_id])):
                data[session_id][subject_id]= normalize_one_session(
                    data[session_id][subject_id], type)

    return data


import awkward as ak

def normalize_one_session(data, type='z_score'):
    '''
    description: Normalizes a jagged EEG array of shape (trials, var_samples, 62, 5)
    param {type}: 'min_max', 'z_score'
    return: list
    '''
    data = ak.Array(data)
    
    # 1. Temporarily flatten the trials and samples together
    # This turns (trials, var_samples, 62, 5) into (total_samples_in_session, 62, 5)
    flat_data = ak.flatten(data, axis=1)
    
    if type == 'min_max':
        # Calculate min/max along the total_samples dimension
        x_min = ak.min(flat_data, axis=0) # Shape becomes (62, 5)
        x_max = ak.max(flat_data, axis=0) # Shape becomes (62, 5)
        _range = x_max - x_min
        
        # Broadcast the (62, 5) stats back across the original jagged array
        ret = (data - x_min) / (_range + 1e-8)
        
    elif type == 'z_score':
        # Calculate mean/std along the total_samples dimension
        x_mean = ak.mean(flat_data, axis=0) # Shape becomes (62, 5)
        x_std = ak.std(flat_data, axis=0)   # Shape becomes (62, 5)
        
        # Broadcast the (62, 5) stats back across the original jagged array
        ret = (data - x_mean) / (x_std + 1e-8)

    return ret.to_list()
def normalize_one_trial(data, type = 'min_max'):
    '''
    description:
    param {type}: min_max, z_score
    return {type}
    '''
    if type == 'min_max':
        _range = np.max(data) - np.min(data)
        ret = (data - np.min(data)) / _range
    elif type == 'z_score':
        x_mean = np.mean(data)
        x_std = np.std(data)
        ret = (data - x_mean) / x_std

    return ret