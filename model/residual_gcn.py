

from einops import einsum, rearrange
from torch import nn
import torch


class SampleAdaptiveAdj(nn.Module):
    """
    Generate per-sample adjacency modulation via Q·K^T attention.

    Inspired by TAHAG (unit_gcn): each sample produces its own electrode
    connectivity graph from its features, blended with a global learned
    adjacency via a learnable α scalar.

    A_final[b] = A_global + α × tanh(Q[b]·K[b]^T / √d)

    α starts at 0 (pure global) and learns how much sample-specificity helps.
    """

    def __init__(self, in_features, num_electrodes, proj_dim=4):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.proj_dim = proj_dim
        # Q and K are 1×1 convolutions (same as TAHAG's conv_a/conv_b)
        self.query = nn.Conv2d(in_features, proj_dim, kernel_size=1, bias=False)
        self.key   = nn.Conv2d(in_features, proj_dim, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))  # starts at 0 = pure global

    def forward(self, x, adj_global):
        """
        Args:
            x:          [B, C, 1, N]  input features (C=bands, N=electrodes)
            adj_global: [N, N]        learned global adjacency
        Returns:
            adj:        [B, N, N]     per-sample adjacency
        """
        Q = self.query(x).squeeze(2)           # [B, d, N]
        K = self.key(x).squeeze(2)             # [B, d, N]
        scale = Q.size(1) ** 0.5
        A_sample = torch.tanh(
            Q.permute(0, 2, 1) @ K / scale     # [B, N, N]
        )
        adj = adj_global.unsqueeze(0) + self.alpha * A_sample
        return torch.relu(adj)                  # [B, N, N] non-negative


class AdjacencyCodebook(nn.Module):
    """
    Discrete codebook for graph adjacency matrices (MIND-EEG Eq. 5-6).

    Quantizes a flattened adjacency matrix to the nearest codebook entry.
    Uses straight-through gradient for the encoder update and a commitment
    loss to pull encoder outputs toward codebook entries.

    Args:
        adj_dim:       size of flattened adjacency (n_electrodes^2)
        num_codes:     number of codebook entries K (default 64, best on SEED-IV)
        commit_weight: λ in the commitment loss term (default 0.25)
    """
    def __init__(self, adj_dim: int, num_codes: int = 64, commit_weight: float = 0.25):
        super().__init__()
        self.commit_weight = commit_weight
        self.codebook = nn.Parameter(torch.randn(num_codes, adj_dim))

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            adj: [N, N] adjacency matrix (the learnable self.adj)
        Returns:
            adj_q:   [N, N] quantized adjacency (straight-through in backward)
            cb_loss: scalar codebook + commitment loss
        """
        N = adj.shape[0]
        a_flat = adj.reshape(1, -1)                      # [1, N*N]

        # Eq. 5: find nearest codebook entry
        dists = torch.cdist(a_flat, self.codebook)       # [1, K]
        idx = dists.argmin(dim=1)                        # [1]
        v_z = self.codebook[idx]                         # [1, N*N]

        # Eq. 6: codebook loss + commitment loss
        cb_loss = (
            (a_flat.detach() - v_z).pow(2).mean()
            + self.commit_weight * (a_flat - v_z.detach()).pow(2).mean()
        )

        # Straight-through estimator: forward uses v_z, backward flows through a_flat
        adj_q = a_flat + (v_z - a_flat).detach()        # [1, N*N]
        return adj_q.reshape(N, N), cb_loss


class MulipleResidualGCN(nn.Module):
    def __init__(self, layers, chan_num, feature_num):
        super().__init__()
        self.chan_num = chan_num
        self.feature_num = feature_num

        # Sample-adaptive adjacency (replaces old RemapAdjacencyMatrix)
        self.adaptive_adj = SampleAdaptiveAdj(
            in_features=feature_num,
            num_electrodes=chan_num,
            proj_dim=4,
        )

        self.residual_gcn_layers = nn.ModuleList()
        for i in range(layers):
            self.residual_gcn_layers.append(ResidualGCN(feature_num=self.feature_num))

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for j in m:
                    if isinstance(j, nn.Linear):
                        nn.init.xavier_uniform_(j.weight, gain=1)

    def forward(self, x, adj=None):
        # adj: [N, N] global learned adjacency
        # Returns per-sample adjacency [B, N, N]
        adj_batch = self.adaptive_adj(x, adj)  # [B, N, N]

        output = []
        output.append(x)
        for i in range(len(self.residual_gcn_layers)):
            input = x
            output.append(self.residual_gcn_layers[i](input, adj_batch))
            x = output[-1]
        out = torch.cat(output, dim=1)
        return out, adj_batch


class ResidualGCN(nn.Module):
    def __init__(self, feature_num):
        """
        Single residual GCN layer with BatchNorm.
        """
        super().__init__()
        self.GConv1 = nn.Conv2d(in_channels=feature_num,
                                out_channels=feature_num,
                                kernel_size=(1, 3),
                                stride=(1, 1),
                                padding=(0, 0),
                                groups=feature_num,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(feature_num)
        self.GConv2 = nn.Conv2d(in_channels=feature_num,
                                out_channels=feature_num,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 1),
                                groups=feature_num,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(feature_num)
        self.ELU = nn.ELU(inplace=False)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, adj_batch):
        """
        Args:
            x:         [B, C, 1, N]   input features
            adj_batch: [B, N, N]      per-sample adjacency
        """
        # Degree-normalize adjacency per sample: L = A * D^{-1}
        # Column-sum (matches original convention)
        deg = adj_batch.sum(dim=-2, keepdim=True).clamp(min=1e-8)  # [B, 1, N]
        adj_normalized = adj_batch / deg                            # [B, N, N]

        residual = x
        x = self.bn2(self.GConv2(self.ELU(self.bn1(self.GConv1(x)))))

        # Batched graph convolution: [B, C, 1, N] × [B, N, N] → [B, C, 1, N]
        y = torch.einsum('bcjk,bkp->bcjp', x, adj_normalized)
        y = self.ELU(torch.add(y, residual))
        return y
