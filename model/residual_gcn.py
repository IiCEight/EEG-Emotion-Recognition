


from einops import einsum, rearrange
from torch import nn
import torch


class MulipleResidualGCN(nn.Module):
    def __init__(self, layers, chan_num, feature_num):
        super().__init__()
        self.chan_num = chan_num
        self.feature_num = feature_num
        self.remap_adj = RemapAdjacencyMatrix(self.chan_num, reduction_ratio=128)
        self.residual_gcn_layers = nn.ModuleList()
        for i in range(layers):
            self.residual_gcn_layers.append(ResidualGCN(feature_num=self.feature_num))

        self.initialize()

    def initialize(self):
        gamma = 0.1
        # self.A = gamma * self.A +  (1 - gamma)*torch.tensor(get_adj_from_standard()).reshape(1, self.chan_num * self.chan_num).float()
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
        # self.A = self.A.to(x.device)
        # A_ds = self.GATENet(self.A)
        adj = self.remap_adj(adj)
        output = []
        output.append(x)
        for i in range(len(self.residual_gcn_layers)):
            input = x
            output.append(self.residual_gcn_layers[i](input, adj))
            x = output[-1]
        out = torch.cat(output, dim=1)
        return out, adj


class RemapAdjacencyMatrix(nn.Module):
    def __init__(self, num_electrodes, reduction_ratio=128):
        super().__init__()
        self.num_electrodes = num_electrodes
        in_channel = num_electrodes * num_electrodes
        self.fc = nn.Sequential(nn.Linear(in_channel, in_channel // reduction_ratio, bias=False),
                                nn.ELU(inplace=False),
                                nn.Linear(in_channel // reduction_ratio, in_channel, bias=False),
                                nn.Tanh(),
                                nn.ReLU(inplace=False))

    def forward(self, adj):
        adj = rearrange(adj, 'row column -> (row column)', row=self.num_electrodes, column = self.num_electrodes)
        adj = self.fc(adj)
        adj = rearrange(adj, '(row column) -> row column', row=self.num_electrodes, column=self.num_electrodes)
        return adj

class ResidualGCN(nn.Module):
    def __init__(self, feature_num):
        """
        dim == 1 for default.
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

        self.ELU = nn.ELU(inplace=False)
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

    def forward(self, x, adj_ds):
        # L = A_ds * D^{-1}
        adj_normalized = einsum(adj_ds, torch.diag(torch.reciprocal(sum(adj_ds))), 'i k, k p -> i p')
        residual = x
        x = self.bn2(self.GConv2(self.ELU(self.bn1(self.GConv1(x)))))
        # discard batch normalization.
        # x = self.GConv2(self.ELU(self.GConv1(x)))
        y = einsum(x, adj_normalized, 'b i j k, k p -> b i j p')
        y = self.ELU(torch.add(y, residual))
        return y
