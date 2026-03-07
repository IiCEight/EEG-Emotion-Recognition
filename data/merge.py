

import awkward as ak
from einops import rearrange
from loguru import logger
import numpy as np


def merge(
    data: ak.Array,
    labels: ak.Array,
):
    """
    input:
        data: ak.Array, shape (subject, session, trial, sample, electrode, feature)
        label: ak.Array, shape (subject, session, trial, sample)
    
    return:
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

    logger.debug("After flattening, data shape: {}, label shape: {}", data.shape, labels.shape)

    data = rearrange(data, "subject session samples electrode feature ->  subject (session samples) electrode feature")
    
    labels = rearrange(labels, 'subject session samples -> subject (session samples)')


    logger.debug("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)

    return data, labels
    


