from loguru import logger
import numpy as np

from data.preprocess.utils import label_process, segment_data


def preprocess_deap(
    data: list,
    labels: list,
    sampling_rate: int,
    num_electrodes: int,
    sample_length: int,
    stride: int,
    label_type: str,
):
    """
    Preprocess DEAP dataset if needed

    return:
    """
    logger.info("Preprocessing DEAP dataset ...")

    # Add any preprocessing steps here if needed

    data, num_feature = segment_data(data, sample_length=sample_length, stride=stride)

    data, labels, num_classes = label_process(
        data, labels, bounds=[5, 5], onehot=True, label_used=[label_type]
    )


    logger.debug(
        "Finished preprocessing DEAP dataset. shape of data: {}, shape of "
        "labels: {},num_electrodes: {}, num_feature: {}, num_classes: {}",
        np.array(data).shape,
        np.array(labels).shape,
        num_electrodes,
        num_feature,
        num_classes,
    )

    return data, labels, num_electrodes, num_feature, num_classes
