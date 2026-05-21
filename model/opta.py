import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Adversarial import DomainAdversarialLoss
from model.classifier import Discriminator
from model.prpl import FeatureExtractor as MlpFeatureExtractor
from model.saber import FeatureExtractor as GcnFeatureExtractor


class CosineClassifier(nn.Module):
    def __init__(self, feat_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f_n = F.normalize(f, dim=1)
        W_n = F.normalize(self.W, dim=1)
        tau = self.log_tau.exp().clamp(min=1e-3)
        return (f_n @ W_n.t()) / tau


class FIFOPool:
    def __init__(self, capacity: int, feat_dim: int, device):
        self.capacity = capacity
        self.feat_dim = feat_dim
        self.buf = torch.zeros(capacity, feat_dim, device=device)
        self.size = 0
        self.ptr = 0

    def push(self, x: torch.Tensor) -> None:
        # x: [k, feat_dim], assumed already detached and L2-normalized
        k = x.size(0)
        for i in range(k):
            self.buf[self.ptr] = x[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def view(self) -> torch.Tensor:
        return self.buf[: self.size]


class OPTA(nn.Module):
    def __init__(
        self,
        num_electrodes: int = 62,
        in_features: int = 5,
        num_classes: int = 3,
        use_gcn: bool = False,
        num_layers: int = 2,
        max_iter: int = 1000,
        pool_capacity: int = 256,
        sinkhorn_lambda: float = 0.05,
        sinkhorn_iters: int = 3,
        triangulation_margin: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_gcn = use_gcn
        self.max_iter = max_iter
        self.pool_capacity = pool_capacity
        self.sinkhorn_lambda = sinkhorn_lambda
        self.sinkhorn_iters = sinkhorn_iters
        self.triangulation_margin = triangulation_margin
        self.feat_dim = 64

        if use_gcn:
            self.feature_extractor = GcnFeatureExtractor(
                num_electrodes=num_electrodes,
                num_feature=in_features,
                layers=num_layers,
                hidden_2=self.feat_dim,
            )
        else:
            self.feature_extractor = MlpFeatureExtractor(
                input_dim=num_electrodes * in_features,
                hidden_1=self.feat_dim,
                hidden_2=self.feat_dim,
            )

        self.classifier = CosineClassifier(feat_dim=self.feat_dim, num_classes=num_classes)
        self.domain_disc = Discriminator(in_feature=self.feat_dim, num_class=2)
        self.dann = DomainAdversarialLoss(self.domain_disc, max_iter=max_iter)

        # Pool is allocated lazily on first forward (we need to know device).
        self.pool: FIFOPool | None = None

        # Cache of last source prototypes for inference.
        self.register_buffer("_last_M_s", torch.zeros(num_classes, self.feat_dim), persistent=False)

    def _ensure_pool(self, device) -> None:
        if self.pool is None:
            self.pool = FIFOPool(self.pool_capacity, self.feat_dim, device)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_label_oh: torch.Tensor,
        epoch: int,
        max_iter: int,
    ):
        self._ensure_pool(source.device)
        f_s = self.feature_extractor(source)
        f_t = self.feature_extractor(target)

        zero = torch.zeros((), device=source.device)
        losses = {
            "src_ce": zero,
            "dann": zero,
            "tgt_ce": zero,
            "tri": zero,
            "xconf": zero,
        }
        diag = {
            "pool_size": float(self.pool.size if self.pool is not None else 0),
            "agree_rate": 0.0,
            "M_t_offdiag_max": 0.0,
        }
        return losses, diag

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        f = self.feature_extractor(x)
        logits = self.classifier(f)
        return logits.argmax(dim=1)

    def get_parameters(self):
        return [
            {"params": self.feature_extractor.parameters(), "lr_mult": 1},
            {"params": self.classifier.parameters(), "lr_mult": 1},
            {"params": self.domain_disc.parameters(), "lr_mult": 1},
        ]
