import math

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.seed import SEED_RGNN_ADJACENCY_MATRIX
from model.classifier import Classifier, Discriminator
from model.grad_reverse import GradientReverse, WarmStartGradientReverseLayer
from model.residual_gcn import MulipleResidualGCN
from utils.graphConstructionFromStandard import get_adj_from_standard


class GatedFusion(nn.Module):
    """
    Attention-gated fusion for two feature maps.

    Given two feature tensors of shape [B, C, H, W], produces a soft
    per-branch scalar gate via global-average-pooled features and a
    learned softmax projection.  The fused output is a convex combination
    of the two inputs (weights always sum to 1).
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 2),
        )

    def forward(self, feat_a, feat_b):
        pool_a = feat_a.mean(dim=[2, 3])
        pool_b = feat_b.mean(dim=[2, 3])

        gate_input = torch.cat([pool_a, pool_b], dim=1)
        gate_weights = F.softmax(self.gate(gate_input), dim=1)

        w_a = gate_weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        w_b = gate_weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)

        return w_a * feat_a + w_b * feat_b


class Saber(nn.Module):
    def __init__(self, num_electrodes=62, in_features=5, num_classes=3, num_layers=2):
        super().__init__()

        self.num_electrodes = num_electrodes
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.grad_reverse_max_iter = 1000

        self.hidden_2 = 64

        self.feature_extractor = FeatureExtractor(
            num_electrodes, in_features, num_layers, hidden_2=self.hidden_2)

        self.class_classifier = Classifier(self.hidden_2, num_classes)
        self.domain_classifier = Discriminator(self.hidden_2)

        self.grad_reverse_layer = WarmStartGradientReverseLayer(
            alpha=1.0, low=0., high=1., max_iters=self.grad_reverse_max_iter, auto_step=True)

    def forward(self, source, target):
        source_feat = self.feature_extractor(source)
        target_feat = self.feature_extractor(target)

        class_output = self.class_classifier(source_feat)

        domain_output_source = self.domain_classifier(self.grad_reverse_layer(source_feat))
        domain_output_target = self.domain_classifier(self.grad_reverse_layer(target_feat))

        return class_output, domain_output_source, domain_output_target, source_feat

    def predict(self, x):
        features = self.feature_extractor(x)
        class_output = self.class_classifier(features)
        return class_output


class FeatureExtractor(nn.Module):
    def __init__(self, num_electrodes, num_feature, layers=2, hidden_2=64):
        super().__init__()
        self.chan_num = num_electrodes
        self.band_num = num_feature
        self.hidden_2 = hidden_2
        self.layers = layers

        self.adj_a = nn.Parameter(torch.tensor(
            get_adj_from_standard()).float(), requires_grad=True)
        self.adj_b = nn.Parameter(torch.tensor(
            get_adj_from_standard()).float(), requires_grad=True)

        # Input normalization (TAHAG-inspired)
        self.data_bn = nn.BatchNorm1d(num_feature)

        self.MRGCN_a = MulipleResidualGCN(layers, self.chan_num, self.band_num)
        self.MRGCN_b = MulipleResidualGCN(layers, self.chan_num, self.band_num)

        mrgcn_out_channels = (layers + 1) * self.band_num

        # Attention-gated fusion
        self.gated_fusion = GatedFusion(channels=mrgcn_out_channels)

        flatten_dim = self.chan_num * mrgcn_out_channels

        # Shared projection
        self.fc1 = nn.Linear(flatten_dim, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)

    def forward(self, x):
        x = x.reshape(x.size(0), 5, 62)
        x = self.data_bn(x)              # normalize across batch
        x = x.unsqueeze(2)

        g_feat_a, _ = self.MRGCN_a(x, self.adj_a)
        g_feat_b, _ = self.MRGCN_b(x, self.adj_b)

        # ---- Gated fusion ----
        g_feat = self.gated_fusion(g_feat_a, g_feat_b)

        # ---- Shared projection ----
        out = self.fc1(g_feat.reshape(g_feat.size(0), -1))
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)

        return out
