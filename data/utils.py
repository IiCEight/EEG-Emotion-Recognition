

import awkward as ak
from einops import rearrange
from loguru import logger
import numpy as np


def merge_for_all_subjects(
    data: ak.Array,
    labels: ak.Array,
    keep_session_dim = False
)-> tuple[np.ndarray, np.ndarray]:
    """
    input:
        data: ak.Array, shape (subject, session, trial, sample, electrode, feature)
        label: ak.Array, shape (subject, session, trial, sample)
    
    return:
        if keep_session_dim:
            data: ak.Array, shape (subject, session, new_sample (trial * sample), electrode, feature)
            label: ak.Array, shape (subject, session, new_sample (trial * sample))
        else:
            data: ak.Array, shape (subject, new_sample (session * trial * sample), electrode, feature)
            label: ak.Array, shape (subject, new_sample (session * trial * sample))
    """
    logger.info(f"Merging data and labels....")

    # 1. flatten the sample dimension, since it is different for each trial.
    data = ak.flatten(data, axis=3)
    labels = ak.flatten(labels, axis=3)

    # 2. Now the shape is perfect the same.
    data = np.array(data)
    labels = np.array(labels)

    if keep_session_dim:
        logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)
        return data, labels

    logger.debug("After flattening, data shape: {}, label shape: {}", data.shape, labels.shape)

    data = rearrange(data, "subject session samples electrode feature ->  subject (session samples) electrode feature")
    
    labels = rearrange(labels, 'subject session samples -> subject (session samples)')


    logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)

    return data, labels

def merge_for_one_subject(
    data: ak.Array,
    labels: ak.Array,
    keep_session_dim = False
)-> tuple[np.ndarray, np.ndarray]:
    """
    input:
        data: ak.Array, shape (session, trial, sample, electrode, feature)
        label: ak.Array, shape (session, trial, sample)
    
    return:
        if keep_session_dim:
            data: ak.Array, shape (session, new_sample (trial * sample), electrode, feature)
            label: ak.Array, shape (session, new_sample (trial * sample))
        else:
            data: ak.Array, shape (new_sample (session * trial * sample), electrode, feature)
            label: ak.Array, shape (new_sample (session * trial * sample))
    """
    logger.info(f"Merging data and labels....")

    # 1. flatten the sample dimension, since it is different for each trial.
    data = ak.flatten(data, axis=2)
    labels = ak.flatten(labels, axis=2)

    # 2. Now the shape is perfect the same.
    data = np.array(data)
    labels = np.array(labels)

    if keep_session_dim:
        logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)
        return data, labels

    logger.debug("After flattening, data shape: {}, label shape: {}", data.shape, labels.shape)

    data = rearrange(data, "session samples electrode feature ->  (session samples) electrode feature")
    
    labels = rearrange(labels, 'session samples -> (session samples)')

    logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)

    return data, labels
    
def split_data(data, labels, split_ratio, random = False):
    """
    input:
        data: ak.Array, shape (session, trial, sample, electrode, feature)
        label: ak.Array, shape (session, trial, sample)
    return:
        train_data: ak.Array, the same as input
        train_label: ak.Array, the same as input
        test_data: ak.Array, the same as input
        test_label: ak.Array, the same as input
    """
    # number of trails for each subject, shape (session)
    num_trials = len(data[0])
    logger.debug(f"Number of trials: {num_trials}")
    num_trails_per_train = int(num_trials * split_ratio)
    mask = [True] * num_trails_per_train + [False] * (num_trials - num_trails_per_train)

    if random:
        np.random.shuffle(mask)

    # Use bool index of ak
    train_data = data[:,mask]
    train_labels = labels[:,mask]
    test_data = data[:,~np.array(mask)]
    test_labels = labels[:,~np.array(mask)]

    return train_data, train_labels, test_data, test_labels
