from loguru import logger

from data.load.load_deap import load_deap
from data.load.load_seed import load_seed
from data.merge_and_split import merge_and_split_deap, merge_and_split_seed
from data.preprocess.preprocess_deap import preprocess_deap
from data.preprocess.preprocess_seed import preprocess_seed
import awkward as ak


def load_data(
    dataset_name: str,
    dataset_path: str,
) -> tuple[ak.Array, ak.Array, int, int, int, int]:
    """
    return:
        data:  list, shape (session, subject, trail, sample, electrode, feature)
        label: list, shape (session, subject, trail, sample)
        num_subjects
        num_electrodes
        num_features
        num_classes
    """
    logger.info(f"Loading dataset {dataset_name} from path {dataset_path}")

    function_map = {
        # "DEAP": load_deap, TODO.
        "SEED": load_seed,
    }

    # Load the data and labels
    return function_map[dataset_name](dataset_path)
