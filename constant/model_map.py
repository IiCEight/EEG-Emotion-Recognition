from constant import CLI_arguments_enum
from model.EEGNet import EEGNet
from model.RGNN_official import SymSimGCNNet


MODEL = {
    CLI_arguments_enum.ModelName.EEGNET: EEGNet,
    CLI_arguments_enum.ModelName.RGNN: SymSimGCNNet,
}

IS_GRAPH_MODEL = {
    CLI_arguments_enum.ModelName.EEGNET: False,
    CLI_arguments_enum.ModelName.RGNN: True,
}