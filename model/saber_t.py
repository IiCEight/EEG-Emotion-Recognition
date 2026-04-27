import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from model.prpl import LabelClassifier, PairLoss, TransferLoss
from model.residual_gcn import MulipleResidualGCN
from utils.graph_construction import get_domain_general_adj, get_weighted_adj


class SaberT(nn.Module):
    """
    Temporal SABER: GRU encodes T consecutive DE windows, then either:
      - use_gcn=True  → per-electrode GRU → residual GCN (TemporalGCNExtractor)
      - use_gcn=False → flat GRU over whole frame → MLP (TemporalMLPExtractor)

    Input shape: [B, T, E, F]
        B = batch size
        T = time steps (e.g. 8 consecutive 1s DE windows)
        E = num_electrodes (62 for SEED)
        F = num_features / frequency bands (5 for SEED)
    """

    def __init__(
        self,
        num_electrodes: int = 62,
        in_features: int = 5,
        time_steps: int = 8,
        num_classes: int = 3,
        num_layers: int = 2,
        max_iter: int = 1000,
        use_gcn: bool = True,
    ):
        super().__init__()
        self.use_gcn = use_gcn
        self.hidden_2 = 64
        self.max_iter = max_iter

        if use_gcn:
            self.feature_extractor = TemporalGCNExtractor(
                num_electrodes=num_electrodes,
                in_features=in_features,
                time_steps=time_steps,
                num_layers=num_layers,
                hidden_2=self.hidden_2,
            )
        else:
            self.feature_extractor = TemporalMLPExtractor(
                num_electrodes=num_electrodes,
                in_features=in_features,
                hidden_2=self.hidden_2,
            )

        self.prpl_classifier = LabelClassifier(
            num_classes=num_classes,
            max_iter=max_iter,
        )
        self.pair_loss = PairLoss(max_iter=max_iter)
        self.transfer_loss = TransferLoss(
            loss_type='dann',
            max_iter=max_iter,
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


class TemporalGCNExtractor(nn.Module):
    """
    Per-electrode GRU encodes T time steps → node feature H.
    Then residual GCN operates over the 62-electrode graph.

    Input:  [B, T, E, F]
    Output: [B, 64]
    """

    GRU_HIDDEN = 16  # hidden size per electrode GRU

    def __init__(self, num_electrodes, in_features, time_steps, num_layers=2, hidden_2=64):
        super().__init__()
        self.E = num_electrodes
        self.F = in_features
        self.T = time_steps
        H = self.GRU_HIDDEN

        # One shared GRU across all electrodes (applied in parallel via batch reshape)
        self.gru = nn.GRU(input_size=in_features, hidden_size=H, batch_first=True)

        _weights_path = Path("cache/electrode_weights.npy")
        if _weights_path.exists():
            _adj_init = get_weighted_adj(str(_weights_path))
        else:
            _adj_init = get_domain_general_adj()
        self.adj = nn.Parameter(torch.tensor(_adj_init).float(), requires_grad=True)

        self.data_bn = nn.BatchNorm1d(H)
        self.mrgcn = MulipleResidualGCN(num_layers, num_electrodes, H)

        mrgcn_out_channels = (num_layers + 1) * H
        flatten_dim = num_electrodes * mrgcn_out_channels

        self.fc1 = nn.Linear(flatten_dim, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)

    def forward(self, x):
        # x: [B, T, E, F]
        B, T, E, n_bands = x.shape

        # Run GRU over time for each electrode independently
        # Reshape to treat each (batch, electrode) pair as one GRU sequence
        x = x.permute(0, 2, 1, 3)              # [B, E, T, n_bands]
        x = x.reshape(B * E, T, n_bands)        # [B*E, T, n_bands]
        _, h = self.gru(x)                      # h: [1, B*E, H]
        h = h.squeeze(0)                        # [B*E, H]
        h = h.reshape(B, E, self.GRU_HIDDEN)    # [B, E, H]

        # Rearrange to match ResidualGCN's expected [B, H, 1, E]
        h = h.permute(0, 2, 1)                  # [B, H, E]
        h = self.data_bn(h)                     # BatchNorm over H channels
        h = h.unsqueeze(2)                      # [B, H, 1, E]

        g_feat, _ = self.mrgcn(h, self.adj)     # [B, (layers+1)*H, 1, E]

        out = self.fc1(g_feat.reshape(B, -1))
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        return out                              # [B, 64]


class TemporalMLPExtractor(nn.Module):
    """
    GRU reads T frames of flattened EEG (E*F each), final hidden state → MLP.

    Input:  [B, T, E, F]
    Output: [B, 64]
    """

    GRU_HIDDEN = 128

    def __init__(self, num_electrodes, in_features, hidden_2=64):
        super().__init__()
        frame_dim = num_electrodes * in_features
        self.gru = nn.GRU(input_size=frame_dim, hidden_size=self.GRU_HIDDEN, batch_first=True)
        self.fc1 = nn.Linear(self.GRU_HIDDEN, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)

    def forward(self, x):
        # x: [B, T, E, F]
        B, T, E, n_bands = x.shape
        x = x.reshape(B, T, E * n_bands)    # [B, T, E*F]
        _, h = self.gru(x)                  # h: [1, B, GRU_HIDDEN]
        h = h.squeeze(0)                    # [B, GRU_HIDDEN]
        out = F.relu(self.fc1(h))           # [B, hidden_2]
        out = F.relu(self.fc2(out))         # [B, hidden_2]
        return out                          # [B, 64]
