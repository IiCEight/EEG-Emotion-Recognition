# Used to parse command line arguments
from enum import Enum

# Define an enumeration for model names.
# This helps in restricting the input arguments to specific model names
class ModelName(str, Enum):
    DGCNN = "DGCNN"
    DANN = "DANN"
    EEGNET = "EEGNet"
    RGNN = "RGNN"
    TAHAG = "TAHAG"
    SABER = "SABER"
    NSAL_DGAT = "NSAL_DGAT"
    PRPL = "PRPL"
    ADANN = "ADANN"
    SABER_T = "SABER_T"
    PCL = "PCL"

class DatasetName(str, Enum):
    DEAP = "DEAP"
    SEED = "SEED"

# Log severity levels
class LevelName(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    ERROR = "ERROR"

# Experimental task types
class TaskTypeName(str, Enum):
    SUBJECT_DEPENDENT = "dep"
    SUBJECT_INDEPENDENT = "indep"

# Experimental task types
class SplitTypeName(str, Enum):
    KFOLD = "kfold"
    LEAVE_ONE_SUBJECT_OUT = "loso"
    TRAIN_TEST_VALIDATION = "ttv"