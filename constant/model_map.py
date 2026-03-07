from constant import CLI_arguments_enum
from model.EEGNet import EEGNet
from model.RGNN_official import SymSimGCNNet
from model.TAHAG_Independent import TAHAG


MODEL = {
    CLI_arguments_enum.ModelName.EEGNET: EEGNet,
    CLI_arguments_enum.ModelName.RGNN: SymSimGCNNet,
    CLI_arguments_enum.ModelName.TAHAG: TAHAG,

}

IS_GRAPH_MODEL = {
    CLI_arguments_enum.ModelName.EEGNET: False,
    CLI_arguments_enum.ModelName.RGNN: True,
    CLI_arguments_enum.ModelName.TAHAG: True,
}