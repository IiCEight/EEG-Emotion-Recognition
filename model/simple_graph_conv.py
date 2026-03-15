
from loguru import logger
from torch import nn
import torch


class SimpleGraphConv(nn.Module):
    """
    SGC-style propagation without adding self-loops:
    H = (D^(-1/2) A D^(-1/2))^K X W.

    Args:
        in_feature: Input feature dimension.
        out_feature: Output feature dimension.
        num_layers: Number of layers K in the SGC.
        bias: Whether to include a bias term in the linear transformation.
    """

    def __init__(self, in_feature, out_feature, num_layers=2, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_feature, out_feature, bias=bias)
        self.bias = bias
        self.num_layers = num_layers
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.linear.weight)
        if self.bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        normalized_adj = normalize_adjacent(adj)
        for _ in range(self.num_layers):
            # For batched input: performs adj[i] @ x[i] for each i.
            x = torch.matmul(normalized_adj, x)
        return self.linear(x)


def normalize_adjacent(adj):
    """
    Symmetric normalization without adding self-loops.

    Supports:
    - adj: (N, N)
    - adj: (B, N, N)
    """
    degree = torch.sum(adj, dim=-1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

    # logger.info("The shape of degree is {}", degree.shape)

    # Broadcast to build D^(-1/2) A D^(-1/2) without explicit diagonal matrices.
    d_left = degree_inv_sqrt.unsqueeze(-1)
    d_right = degree_inv_sqrt.unsqueeze(-2)
    normalized_adj = d_left * adj * d_right
    return normalized_adj