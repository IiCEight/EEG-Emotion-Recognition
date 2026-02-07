from constant import CLI_arguments_enum
from model.EEGNet import EEGNet


MODEL = {
    CLI_arguments_enum.ModelName.EEGNET: EEGNet,
}