from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from model.prpl import LabelClassifier, PairLoss, TransferLoss
from model.prpl import FeatureExtractor as MlpFeatureExtractor
from model.residual_gcn import MulipleResidualGCN
from utils.graphConstructionFromStandard import get_adj_from_standard
from utils.graph_construction import get_domain_general_adj, get_weighted_adj



class Saber(nn.Module):
    def __init__(self, num_electrodes=62, in_features=5, num_classes=3, num_layers=2, max_iter=1000, use_gcn=False):
        super().__init__()

        self.num_electrodes = num_electrodes
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_2 = 64
        self.use_gcn = use_gcn
        self.max_iter = max_iter

        if use_gcn:
            self.feature_extractor = FeatureExtractor(
                num_electrodes=num_electrodes,
                num_feature=in_features,
                layers=num_layers,
                hidden_2=self.hidden_2,
            )
        else:
            self.feature_extractor = MlpFeatureExtractor(
                input_dim=num_electrodes * in_features,
                hidden_1=self.hidden_2,
                hidden_2=self.hidden_2,
            )

        self.prpl_classifier = LabelClassifier(
            num_classes=num_classes,
            max_iter=self.max_iter,
        )
        self.pair_loss = PairLoss(max_iter=self.max_iter)
        self.transfer_loss = TransferLoss(
            loss_type='dann',
            max_iter=self.max_iter,
            hidden_1=self.hidden_2,
        )

    def forward(self, source, target, source_label):
        source_feat = self.feature_extractor(source)
        target_feat = self.feature_extractor(target)
        batch_size = source_feat.size(0)

        self.prpl_classifier.update_P(source_feat, source_label)

        source_logits = self.prpl_classifier(source_feat)
        target_logits = self.prpl_classifier(target_feat)

        clf_loss, cluster_loss = self.pair_loss(source_label, source_logits, target_logits)
        p_loss = torch.norm(
            torch.matmul(self.prpl_classifier.P.T, self.prpl_classifier.P)
            - torch.eye(self.hidden_2, device=source.device),
            'fro',
        )
        trans_loss = self.transfer_loss(
            source_feat + 0.005 * torch.randn((batch_size, self.hidden_2), device=source_feat.device),
            target_feat + 0.005 * torch.randn((batch_size, self.hidden_2), device=target_feat.device),
        )
        return clf_loss, cluster_loss, p_loss, trans_loss

    def epoch_end_hook(self, epoch, source_features, source_labels):
        self.pair_loss.update_threshold(epoch)
        features = self.feature_extractor(source_features)
        self.prpl_classifier.update_cluster_label(features, source_labels)

    def predict(self, x):
        feature = self.feature_extractor(x)
        return self.prpl_classifier.predict(feature)


class FeatureExtractor(nn.Module):
    def __init__(self, num_electrodes, num_feature, layers=2, hidden_2=64):
        super().__init__()
        self.chan_num = num_electrodes
        self.band_num = num_feature
        self.hidden_2 = hidden_2
        self.layers = layers

        _weights_path = Path("cache/electrode_weights.npy")
        # if _weights_path.exists():
        #     _adj_init = get_weighted_adj(str(_weights_path))
        # else:
        #     _adj_init = get_domain_general_adj()
        logger.info("Use direct distance as adjeceny matrix for GCN.")
        _adj_init = get_adj_from_standard()

        self.adj = nn.Parameter(torch.tensor(_adj_init).float(), requires_grad=True)

        self.data_bn = nn.BatchNorm1d(num_feature)
        self.mrgcn = MulipleResidualGCN(layers, self.chan_num, self.band_num)

        mrgcn_out_channels = (layers + 1) * self.band_num
        flatten_dim = self.chan_num * mrgcn_out_channels

        self.fc1 = nn.Linear(flatten_dim, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)

    def forward(self, x):
        x = x.reshape(x.size(0), self.band_num, self.chan_num)
        x = self.data_bn(x)
        x = x.unsqueeze(2)

        g_feat, _ = self.mrgcn(x, self.adj)

        out = self.fc1(g_feat.reshape(g_feat.size(0), -1))
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        return out
