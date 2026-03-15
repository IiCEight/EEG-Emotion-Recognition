from constant import CLI_arguments_enum
from model.EEGNet import EEGNet
from model.RGNN import RGNN
from model.RGNN_official import SymSimGCNNet
from model.TAHAG_Independent import TAHAG
from model.saber import Saber


MODEL = {
    CLI_arguments_enum.ModelName.EEGNET: EEGNet,
    CLI_arguments_enum.ModelName.RGNN: RGNN,
    CLI_arguments_enum.ModelName.TAHAG: TAHAG,
    CLI_arguments_enum.ModelName.SABER: Saber,

}

IS_GRAPH_MODEL = {
    CLI_arguments_enum.ModelName.EEGNET: False,
    CLI_arguments_enum.ModelName.RGNN: True,
    CLI_arguments_enum.ModelName.TAHAG: True,
}