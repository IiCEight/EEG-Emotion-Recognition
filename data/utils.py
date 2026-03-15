

import awkward as ak
from einops import rearrange
from loguru import logger
import numpy as np


def merge_for_all_subjects(
    data: list,
    labels: list,
    merge_subject_dim = True
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

    data = ak.Array(data)
    labels = ak.Array(labels)

    # 1. flatten the sample dimension, since it is different for each trial.
    data = ak.flatten(data, axis=2)
    labels = ak.flatten(labels, axis=2)

    # 2. Now the shape is perfect the same.
    data = np.array(data)
    labels = np.array(labels)

    if not merge_subject_dim:
        logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)
        return data, labels

    if merge_subject_dim:
        data = rearrange(data, "subject samples electrode feature -> (subject samples) electrode feature")
        labels = rearrange(labels, "subject samples -> (subject samples)")

    logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)

    return data, labels

def merge_for_one_subject(
    data: list,
    labels: list
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

    data = ak.Array(data)
    labels = ak.Array(labels)

    # 1. flatten the sample dimension, since it is different for each trial.
    data = ak.flatten(data, axis=1)
    labels = ak.flatten(labels, axis=1)

    # 2. Now the shape is perfect the same.
    data = np.array(data)
    labels = np.array(labels)

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
    train_data = data[mask].to_list()
    train_labels = labels[mask].to_list()
    test_data = data[~mask].to_list()
    test_labels = labels[~mask].to_list()

    # convert back to list
    
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
