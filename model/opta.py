import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Adversarial import DomainAdversarialLoss
from model.classifier import Discriminator
from model.prpl import FeatureExtractor as MlpFeatureExtractor
from model.saber import FeatureExtractor as GcnFeatureExtractor


class _LabelSmoothingCE(nn.Module):
    def __init__(self, num_classes: int, epsilon: float = 0.0005):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        weight = logits.new_ones(logits.size()) * self.epsilon / (logits.size(-1) - 1.0)
        weight.scatter_(-1, target.unsqueeze(-1), 1.0 - self.epsilon)
        return (-weight * log_prob).sum(dim=-1).mean()


def source_prototypes(f_s: torch.Tensor, y_s_oh: torch.Tensor) -> torch.Tensor:
    counts = y_s_oh.sum(0).clamp(min=1.0).unsqueeze(1)  # [C, 1]
    M_s = (y_s_oh.t() @ f_s) / counts                   # [C, D]
    return F.normalize(M_s, dim=1)


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
        self.domain_disc = Discriminator(in_feature=self.feat_dim, num_class=1)
        self.dann = DomainAdversarialLoss(self.domain_disc, max_iter=max_iter)
        self.smooth_ce = _LabelSmoothingCE(num_classes=num_classes)

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

        # Source prototypes (cached for inference; per-batch, on-graph)
        M_s = source_prototypes(f_s, source_label_oh)
        self._last_M_s.copy_(M_s.detach())

        # Source CE on cosine classifier
        logits_s = self.classifier(f_s)
        src_label = source_label_oh.argmax(dim=1)
        loss_src_ce = self.smooth_ce(logits_s, src_label)

        # DANN with feature noise (matches SABER/PRPL pattern)
        noise = 0.005
        loss_dann = self.dann(
            f_s + noise * torch.randn_like(f_s),
            f_t + noise * torch.randn_like(f_t),
        )

        zero = torch.zeros((), device=source.device)
        losses = {
            "src_ce": loss_src_ce,
            "dann": loss_dann,
            "tgt_ce": zero,
            "tri": zero,
            "xconf": zero,
        }
        diag = {
            "pool_size": float(self.pool.size),
            "agree_rate": 0.0,
            "M_t_offdiag_max": 0.0,
        }
        return losses, diag

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        f = self.feature_extractor(x)
        # Stage 2: use source prototypes (target prototypes not yet implemented).
        f_n = F.normalize(f, dim=1)
        M = self._last_M_s
        return (f_n @ M.t()).argmax(dim=1)

    def get_parameters(self):
        return [
            {"params": self.feature_extractor.parameters(), "lr_mult": 1},
            {"params": self.classifier.parameters(), "lr_mult": 1},
            {"params": self.domain_disc.parameters(), "lr_mult": 1},
        ]
