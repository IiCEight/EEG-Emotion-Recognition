import math

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.seed import SEED_RGNN_ADJACENCY_MATRIX
from model.classifier import Classifier, Discriminator
from model.grad_reverse import GradientReverse, WarmStartGradientReverseLayer
from model.graph_attention import CBAMBlock
from model.residual_gcn import MulipleResidualGCN
from model.simple_graph_conv import SimpleGraphConv
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


class AuxiliaryClassifier(nn.Module):
    """Lightweight binary classifier for emotion-polarity specialization."""

    def __init__(self, in_features, num_classes=2):
        super().__init__()
        hidden = in_features // 2
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class Saber(nn.Module):
    def __init__(self, num_electrodes=62, in_features=5, num_classes=3, num_layers=2):
        super().__init__()

        self.num_electrodes = num_electrodes
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.conv_out_dim = 16
        self.grad_reverse_max_iter = 1000

        self.hidden_1 = 256
        self.hidden_2 = 64

        self.feature_extractor = FeatureExtractor(
            num_electrodes, in_features, num_layers, self.hidden_1, self.hidden_2, num_classes)

        self.class_classifier = Classifier(self.hidden_2, num_classes)
        self.domain_classifier = Discriminator(self.hidden_2)

        self.grad_reverse_layer = WarmStartGradientReverseLayer(
            alpha=1.0, low=0., high=1., max_iters=self.grad_reverse_max_iter, auto_step=True)

        # Auxiliary polarity heads
        self.aux_classifier_a = AuxiliaryClassifier(self.hidden_2, num_classes=2)
        self.aux_classifier_b = AuxiliaryClassifier(self.hidden_2, num_classes=2)

    def forward(self, source, target):
        source_fused, source_branch_a, source_branch_b = self.feature_extractor(source, return_branches=True)
        target_fused, _, _ = self.feature_extractor(target, return_branches=True)

        class_output = self.class_classifier(source_fused)

        domain_output_source = self.domain_classifier(self.grad_reverse_layer(source_fused))
        domain_output_target = self.domain_classifier(self.grad_reverse_layer(target_fused))

        aux_a_logits = self.aux_classifier_a(source_branch_a)
        aux_b_logits = self.aux_classifier_b(source_branch_b)

        return (class_output, domain_output_source, domain_output_target,
                aux_a_logits, aux_b_logits,
                source_branch_a, source_branch_b,
                source_fused)   # for contrastive loss

    def predict(self, x):
        features = self.feature_extractor(x, return_branches=False)
        class_output = self.class_classifier(features)
        return class_output


class FeatureExtractor(nn.Module):
    def __init__(self, num_electrodes, num_feature, layers=2, hidden_1=256, hidden_2=64, class_nums=3):
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

        self.CBAM_a = CBAMBlock(channel=mrgcn_out_channels, reduction=4, kernel_size=3)
        self.CBAM_b = CBAMBlock(channel=mrgcn_out_channels, reduction=4, kernel_size=3)

        flatten_dim = self.chan_num * mrgcn_out_channels

        # Shared projection
        self.fc1 = nn.Linear(flatten_dim, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)

        # Per-branch projections (flatten GCN features for aux/ortho)
        self.branch_proj_a = nn.Sequential(
            nn.Linear(flatten_dim, hidden_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_2, hidden_2),
            nn.ReLU(inplace=True),
        )
        self.branch_proj_b = nn.Sequential(
            nn.Linear(flatten_dim, hidden_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_2, hidden_2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, return_branches=False):
        x = x.reshape(x.size(0), 5, 62)
        x = self.data_bn(x)              # normalize across batch
        x = x.unsqueeze(2)

        g_feat_a, g_adj_a = self.MRGCN_a(x, self.adj_a)
        g_feat_b, g_adj_b = self.MRGCN_b(x, self.adj_b)

        # ---- CBAM per-branch ----
        g_feat_a = self.CBAM_a(g_feat_a)
        g_feat_b = self.CBAM_b(g_feat_b)

        # ---- Gated fusion ----
        g_feat = self.gated_fusion(g_feat_a, g_feat_b)

        # ---- Shared projection ----
        out = self.fc1(g_feat.reshape(g_feat.size(0), -1))
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)

        if return_branches:
            flat_a = g_feat_a.reshape(g_feat_a.size(0), -1)
            flat_b = g_feat_b.reshape(g_feat_b.size(0), -1)
            branch_a_feat = self.branch_proj_a(flat_a)
            branch_b_feat = self.branch_proj_b(flat_b)
            return out, branch_a_feat, branch_b_feat

        return out
