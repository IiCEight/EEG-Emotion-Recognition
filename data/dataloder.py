from loguru import logger

from data.load.load_deap import load_deap
from data.preprocess.preprocess_deap import preprocess_deap


def get_data(dataset_name: str, dataset_path: str, sample_length: int, stride: int, label_type: str = None):
    logger.info(f"Loading dataset {dataset_name} from path {dataset_path}")

    function_map = {
        "DEAP": [load_deap, preprocess_deap],
    }

    # Load the data and labels
    data, labels, sampling_rate, num_electrodes = function_map[dataset_name][0](
        dataset_path
    )
    # Preprocess the data
    data, labels, num_electrodes, num_features, num_classes = function_map[
        dataset_name
    ][1](data, labels, sampling_rate, num_electrodes, sample_length, stride, label_type)

    return (
        data,
        labels,
        num_electrodes,
        num_features,
        num_classes,
    )
