from loguru import logger

from data.load.load_deap import load_deap
from data.load.load_seed import load_seed
from data.merge_and_split import merge_and_split, merge_and_split_deap, merge_and_split_seed
from data.preprocess.preprocess_deap import preprocess_deap
from data.preprocess.preprocess_seed import preprocess_seed


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
        "DEAP": [load_deap, preprocess_deap, merge_and_split_deap],
        "SEED": [load_seed, preprocess_seed, merge_and_split_seed],
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
        function_map[dataset_name][2](data, labels, num_classes, task_type)
    )

    return (
        split_dataset,
        num_subjects,
        num_electrodes,
        num_features,
        num_classes,
    )
