"""
PCL-TDGCN with Saber's FeatureExtractor (MulipleResidualGCN) replacing the original MHGCN Encoder.
All PCL logic (memory banks, DANN, prototypical contrastive learning) is unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from model.saber import FeatureExtractor


class Discriminator(nn.Module):
    def __init__(self, hidden_1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class _SaberEncoder(nn.Module):
    """Wraps FeatureExtractor to match the (features, aux) tuple contract of the original Encoder."""

    def __init__(self, in_planes, layers, hidden_2):
        super().__init__()
        band_num, chan_num = in_planes[0], in_planes[1]
        self.fe = FeatureExtractor(
            num_electrodes=chan_num,
            num_feature=band_num,
            layers=layers,
            hidden_2=hidden_2,
        )
        # Expose fc2 so training_pcl.py can read fc2.out_features
        self.fc2 = self.fe.fc2

    def forward(self, x):
        return self.fe(x), []


class ClassClassifier(nn.Module):
    def __init__(self, hidden_2, num_cls):
        super().__init__()
        self.classifier = nn.Linear(hidden_2, num_cls)

    def forward(self, x):
        return self.classifier(x)


class PCL_SABER(nn.Module):
    def __init__(self, in_planes=[5, 62], layers=2, hidden_1=256,
                 hidden_2=64, num_of_class=3, device='cuda:0',
                 source_num=3944, target_num=851):
        super().__init__()

        self.encoder = _SaberEncoder(in_planes=in_planes, layers=layers, hidden_2=hidden_2)
        self.cls_classifier = ClassClassifier(hidden_2=hidden_2, num_cls=num_of_class)

        print(f"number of source samples {source_num}")
        print(f"number of target samples {target_num}")

        self.source_f_bank = torch.zeros(source_num, hidden_2)
        self.target_f_bank = torch.zeros(target_num, hidden_2)
        self.source_score_bank = torch.zeros(source_num, num_of_class).to(device)
        self.target_score_bank = torch.zeros(target_num, num_of_class).to(device)
        self.source_label_bank = torch.full((source_num,), -1, dtype=torch.long)

        self.num_of_class = num_of_class
        self.ema_factor = 0.8
        self.tem = 1
        self.device = device

    def forward(self, source, target, source_label,
                source_index, target_index, current_epoch, max_epochs):
        source_f, _ = self.encoder(source)
        target_f, _ = self.encoder(target)

        source_predict = self.cls_classifier(source_f)
        target_predict = self.cls_classifier(target_f)

        source_label_feature = F.softmax(source_predict, dim=1)
        target_label_feature = F.softmax(target_predict, dim=1)

        src_sim, src_prototype = self._get_source_similar(source_f, source_label_feature, source_index)
        tgt_sim, tgt_prototype, tat_cluster_label = self._get_target_similar(
            target_f, target_label_feature, target_index, src_prototype, current_epoch, max_epochs
        )

        s2t_pro = self._get_st_similar(source_f, tgt_prototype)
        t2s_pro = self._get_st_similar(target_f, src_prototype)
        s2s_pro = self._get_st_similar(source_f, src_prototype)
        t2t_pro = self._get_st_similar(target_f, tgt_prototype)

        return (source_predict, source_f, target_predict, target_f,
                [], [],
                src_sim, tgt_sim, tat_cluster_label,
                s2t_pro, t2s_pro, s2s_pro, t2t_pro)

    def _get_source_similar(self, feature_source_f, source_label_feature, source_index):
        self.eval()
        output_f = F.normalize(feature_source_f, p=2, dim=1)

        self.source_f_bank[source_index] = output_f.detach().clone().cpu()
        self.source_score_bank[source_index] = source_label_feature.detach().clone()

        prototype_class = []
        for class_id in range(self.num_of_class):
            pred_labels = torch.argmax(self.source_score_bank, dim=1).cpu()
            source_feature = self.source_f_bank[pred_labels == class_id]
            if source_feature.size(0) > 0:
                prototype = source_feature.mean(dim=0)
            else:
                prototype = torch.zeros(output_f.size(1), device=source_feature.device)
            prototype_class.append(prototype)

        prototypes = torch.stack(prototype_class)

        src_sim = torch.mm(
            output_f.to(self.device),
            F.normalize(prototypes.to(self.device), p=2, dim=1).T
        ) / self.tem

        return src_sim, prototypes

    def _get_target_similar(self, feature_target_f, target_label_feature,
                            target_index, src_prototype, current_epoch, max_epochs):
        self.eval()
        f = F.normalize(feature_target_f, p=2, dim=1)

        self.target_f_bank[target_index] = f.detach().clone().cpu()
        self.target_score_bank[target_index] = target_label_feature.detach().clone()

        output = self.target_f_bank.to(self.device)
        scores = self.target_score_bank
        aggregated_scores = scores.max(dim=1)[0]
        num_samples = len(aggregated_scores)

        k = int(num_samples * 0.3)
        _, top_indices = torch.topk(aggregated_scores, k)
        output_f = output[top_indices]

        kmeans = KMeans(n_clusters=self.num_of_class, random_state=0, n_init="auto")
        pool = output_f.cpu().detach().numpy()
        pool = np.unique(pool, axis=0)
        if len(pool) < self.num_of_class:
            rng = np.random.default_rng(0)
            while len(pool) < self.num_of_class:
                pool = np.vstack([pool, pool[rng.integers(len(pool))] + rng.standard_normal(pool.shape[1]) * 1e-4])
        kmeans.fit(pool)
        prototype = torch.tensor(kmeans.cluster_centers_, device=self.device)

        tgt_sim = torch.mm(
            F.normalize(f, p=2, dim=1),
            F.normalize(prototype, p=2, dim=1).T
        ) / self.tem

        target_predict = F.softmax(tgt_sim, dim=1)
        tar_label = torch.argmax(target_predict, dim=1)

        return tgt_sim, prototype, tar_label

    def _get_st_similar(self, feature, prototypes):
        if prototypes.numel() == 0:
            return torch.zeros((feature.size(0), 3), device=feature.device)

        feature = F.normalize(feature, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        st_sim = torch.mm(
            feature.to(self.device),
            prototypes.to(self.device).T
        ) / self.tem

        return F.softmax(st_sim, dim=1)

    def get_init_banks(self, source, source_index):
        self.eval()
        with torch.no_grad():
            source_f, _ = self.encoder(source)
            source_predict = self.cls_classifier(source_f)
            source_label_feature = F.softmax(source_predict, dim=1)
            self.source_f_bank[source_index] = F.normalize(source_f).detach().clone().cpu()
            self.source_score_bank[source_index] = source_label_feature.detach().clone()

    def get_init_banks_tgt(self, tgt, tgt_index):
        self.eval()
        with torch.no_grad():
            tgt_f, _ = self.encoder(tgt)
            tgt_predict = self.cls_classifier(tgt_f)
            tgt_label_feature = F.softmax(tgt_predict, dim=1)
            self.target_f_bank[tgt_index] = F.normalize(tgt_f).detach().clone().cpu()
            self.target_score_bank[tgt_index] = tgt_label_feature.detach().clone()

    def target_predict(self, feature_target):
        self.eval()
        with torch.no_grad():
            target_f, _ = self.encoder(feature_target)
            target_predict = self.cls_classifier(target_f)
            return F.softmax(target_predict, dim=1)
