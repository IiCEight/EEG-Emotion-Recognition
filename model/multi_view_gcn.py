import torch
import torch.nn as nn
import torch.nn.functional as F

from model.residual_gcn import MulipleResidualGCN
from utils.graphConstructionFromStandard import get_adj_from_standard


class MultiViewGCN(nn.Module):
    """Two-branch GCN with independent adjacencies and switchable fusion.

    Args:
        num_electrodes: Number of EEG electrodes. Default 62.
        in_features: Number of frequency bands per electrode. Default 5.
        num_layers: Residual GCN layers per branch. Default 2.
        fusion: "concat" or "attn".
    """

    def __init__(
        self,
        num_electrodes: int = 62,
        in_features: int = 5,
        num_layers: int = 2,
        fusion: str = "concat",
    ):
        super().__init__()
        assert fusion in ("concat", "attn"), f"fusion must be 'concat' or 'attn', got {fusion!r}"
        self.fusion = fusion
        self.num_electrodes = num_electrodes
        self.in_features = in_features

        # View 0: physical electrode distance adjacency (same init as Saber).
        # adj_0 is fixed to the 62-electrode SEED layout; num_electrodes must be 62.
        adj0 = torch.tensor(get_adj_from_standard()).float()
        self.adj_0 = nn.Parameter(adj0, requires_grad=True)

        # View 1: uniform adjacency in [0, 1], data-driven (same value range as adj_0)
        adj1 = torch.empty(num_electrodes, num_electrodes).uniform_(0, 1)
        self.adj_1 = nn.Parameter(adj1, requires_grad=True)

        self.gcn_0 = MulipleResidualGCN(num_layers, num_electrodes, in_features)
        self.gcn_1 = MulipleResidualGCN(num_layers, num_electrodes, in_features)

        self.data_bn = nn.BatchNorm1d(in_features)

        # (num_layers + 1) * in_features channels after MulipleResidualGCN
        view_flat = num_electrodes * (num_layers + 1) * in_features  # 62*3*5 = 930

        if fusion == "concat":
            self.fc1 = nn.Linear(view_flat * 2, 64)   # 1860 -> 64
            self.fc2 = nn.Linear(64, 64)
        else:  # "attn"
            self.fc_v0 = nn.Linear(view_flat, 64)      # per-view projection
            self.fc_v1 = nn.Linear(view_flat, 64)
            self.fc_attn = nn.Linear(view_flat * 2, 2) # attention logits
            self.fc2 = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 62, 5]  (electrodes × bands) — same input format as Saber FeatureExtractor
        B = x.size(0)
        x = x.reshape(B, self.in_features, self.num_electrodes)  # [B, 5, 62]
        x = self.data_bn(x)
        x = x.unsqueeze(2)  # [B, 5, 1, 62]

        g0, _ = self.gcn_0(x, self.adj_0)   # [B, 15, 1, 62]
        g1, _ = self.gcn_1(x, self.adj_1)   # [B, 15, 1, 62]

        v0 = g0.reshape(B, -1)  # [B, 930]
        v1 = g1.reshape(B, -1)  # [B, 930]

        cat = torch.cat([v0, v1], dim=1)  # [B, 1860]

        if self.fusion == "concat":
            out = F.relu(self.fc1(cat))   # [B, 64]
        else:
            v0_emb = self.fc_v0(v0)       # [B, 64]
            v1_emb = self.fc_v1(v1)       # [B, 64]
            attn_w = F.softmax(self.fc_attn(cat), dim=1)  # [B, 2]
            out = attn_w[:, 0:1] * v0_emb + attn_w[:, 1:2] * v1_emb  # [B, 64]
            out = F.relu(out)

        out = F.relu(self.fc2(out))  # [B, 64]
        return out
