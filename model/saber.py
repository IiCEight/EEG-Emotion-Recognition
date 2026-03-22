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

    def forward(self, source, target):
        source_f = self.feature_extractor(source)
        target_f = self.feature_extractor(target)

        class_output = self.class_classifier(source_f)

        domain_output_source = self.domain_classifier(self.grad_reverse_layer(source_f))
        domain_output_target = self.domain_classifier(self.grad_reverse_layer(target_f))    

        return class_output, domain_output_source, domain_output_target
    
    def predict(self, x):
        features = self.feature_extractor(x)
        class_output = self.class_classifier(features)
        return class_output


class FeatureExtractor(nn.Module):
    def __init__(self, num_electrodes, num_feature, layers=2, hidden_1=256, hidden_2=64, class_nums=3):
        super().__init__()
        self.chan_num = num_electrodes
        self.band_num = num_feature
        self.adj_a = nn.Parameter(torch.tensor(
            get_adj_from_standard()).float(), requires_grad=True)
        self.adj_b = nn.Parameter(torch.tensor(
            get_adj_from_standard()).float(), requires_grad=True)

        self.MRGCN_a = MulipleResidualGCN(layers, self.chan_num, self.band_num)

        self.MRGCN_b = MulipleResidualGCN(layers, self.chan_num, self.band_num)

        self.CBAM = CBAMBlock(channel=(layers + 1) *
                              self.band_num, reduction=4, kernel_size=3)
        self.fc1 = nn.Linear(self.chan_num * (layers + 1)
                             * self.band_num, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        # x = rearrange(x, 'b chan feature -> b feature chan', chan= 62, feature = 5)
        x = x.reshape(x.size(0), 5, 62)
        x = x.unsqueeze(2)
        # logger.info('Encoder input shape after reshape: {}', x.shape)

        g_feat_a, g_adj_a = self.MRGCN_a(x, self.adj_a)
        g_feat_b, g_adj_b = self.MRGCN_b(x, self.adj_b)
        # g_feat, g_adj = self.MRGCN(x)

        g_feat = (g_feat_a + g_feat_b) / 2.0
        g_adj = (g_adj_a + g_adj_b) / 2.0

        g_feat, ca, sa = self.CBAM(g_feat)
        out = self.fc1(g_feat.reshape(g_feat.size(0), -1))
        # logger.info("out shape after fc1: {}", out.shape)
        out = F.relu(out)
        # out = self.dropout1(out)
        out = self.fc2(out)
        out = F.relu(out)
        # out = self.dropout2(out)
        return out
