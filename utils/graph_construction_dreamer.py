import numpy as np
from utils.graphConstructionFromStandard import (
    format_adj_matrix_from_standard,
    STANDARD_1005_CHANNEL_LOCATION_DICT,
)

DREAMER_CHANNEL_LIST = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]


def get_dreamer_adj_from_standard() -> np.ndarray:
    return format_adj_matrix_from_standard(
        DREAMER_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT
    )
