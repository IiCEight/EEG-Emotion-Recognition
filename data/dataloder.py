from loguru import logger

from data.load.load_deap import load_deap
from data.merge_and_split import merge_and_split
from data.preprocess.preprocess_deap import preprocess_deap


def load_data(
    dataset_name: str,
    dataset_path: str,
    sample_length: int,
    stride: int,
    task_type: str,
    split_type: str,
    label_type: str = None,
) -> tuple[list, list, int, int, int]:
    logger.info(f"Loading dataset {dataset_name} from path {dataset_path}")

    function_map = {
        "DEAP": [load_deap, preprocess_deap],
    }

    # Load the data and labels
    data, labels, sampling_rate,num_subjects, num_electrodes = function_map[dataset_name][0](
        dataset_path
    )
    # Preprocess the data
    data, labels, num_electrodes, num_features, num_classes = function_map[
        dataset_name
    ][1](data, labels, sampling_rate, num_electrodes, sample_length, stride, label_type)

    # split the data into training, testing, validation sets according to the task type
    split_dataset = (
        merge_and_split(data, labels, num_classes, task_type)
    )

    return (
        split_dataset,
        num_subjects,
        num_electrodes,
        num_features,
        num_classes,
    )
