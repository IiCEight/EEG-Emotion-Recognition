# Used to parse command line arguments
from enum import Enum

# Define an enumeration for model names.
# This helps in restricting the input arguments to specific model names
class ModelName(str, Enum):
    DGCNN = "DGCNN"
    DANN = "DANN"

class DatasetName(str, Enum):
    DEAP = "DEAP"
    SEED = "SEED"

# Log severity levels
class LevelName(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    ERROR = "ERROR"