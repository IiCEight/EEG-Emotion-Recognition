from constant import CLI_arguments_enum
from model.EEGNet import EEGNet
from model.RGNN import RGNN
from model.RGNN_official import SymSimGCNNet
from model.TAHAG_Independent import TAHAG
from model.saber import Saber
from model.NSAL_DGAT import Domain_adaption_model
from model.prpl import PRPL
from model.adann import ADANN
from model.saber_t import SaberT


MODEL = {
    CLI_arguments_enum.ModelName.EEGNET: EEGNet,
    CLI_arguments_enum.ModelName.RGNN: RGNN,
    CLI_arguments_enum.ModelName.TAHAG: TAHAG,
    CLI_arguments_enum.ModelName.SABER: Saber,
    CLI_arguments_enum.ModelName.NSAL_DGAT: Domain_adaption_model,
    CLI_arguments_enum.ModelName.PRPL: PRPL,
    CLI_arguments_enum.ModelName.ADANN: ADANN,
    CLI_arguments_enum.ModelName.SABER_T: SaberT,

}

IS_GRAPH_MODEL = {
    CLI_arguments_enum.ModelName.EEGNET: False,
    CLI_arguments_enum.ModelName.RGNN: True,
    CLI_arguments_enum.ModelName.TAHAG: True,
    CLI_arguments_enum.ModelName.NSAL_DGAT: True,
    CLI_arguments_enum.ModelName.PRPL: False,
    CLI_arguments_enum.ModelName.ADANN: False,
    CLI_arguments_enum.ModelName.SABER_T: True,
}