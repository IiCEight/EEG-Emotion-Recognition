from typing import Optional

import torch

from reference.NSAL_DGAT import DAANLoss as _RefDAANLoss
from reference.NSAL_DGAT import Discriminator
from reference.NSAL_DGAT import Domain_adaption_model


class DAANLoss(_RefDAANLoss):
    """Compatibility wrapper with robust source/target splitting for unequal batch sizes."""

    def get_global_adversarial_result(self, f_s: torch.Tensor, f_t: torch.Tensor):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.global_classifiers(f)

        src_batch = f_s.size(0)
        tgt_batch = f_t.size(0)
        d_s = d[:src_batch]
        d_t = d[src_batch:src_batch + tgt_batch]

        d_label_s = torch.ones((src_batch, 1), device=f_s.device)
        d_label_t = torch.zeros((tgt_batch, 1), device=f_t.device)
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class NSALDGAT(Domain_adaption_model):
    def __init__(
        self,
        num_electrodes: int = 62,
        in_features: int = 5,
        num_classes: int = 3,
        layers: int = 2,
        hidden_2: int = 64,
        dropout: float = 0.25,
        domain_adaptation: bool = False,
        source_num: int = 8192,
        device: str = "cpu",
    ):
        super().__init__(
            channels=num_electrodes,
            feature_dim=in_features,
            num_of_class=num_classes,
            layers=layers,
            hidden_2=hidden_2,
            device=device,
            source_num=source_num,
        )
        self.domain_adaptation = domain_adaptation
        self._device_name = device

    def reset_source_bank(self, source_num: int, device: Optional[torch.device] = None) -> None:
        score_device = device if device is not None else self.source_score_bank.device
        hidden_dim = self.source_f_bank.shape[1]
        self.source_f_bank = torch.randn(source_num, hidden_dim, device=score_device)
        self.source_score_bank = torch.randn(source_num, self.num_of_class, device=score_device)

    def get_target_labels(self, feature_source_f, source_label_feature, source_index, feature_target_f):
        source_index = source_index.long()
        output_f = torch.nn.functional.normalize(feature_source_f)
        self.source_f_bank[source_index] = output_f.detach().clone()
        self.source_score_bank[source_index] = source_label_feature.detach().clone()

        output_f_ = torch.nn.functional.normalize(feature_target_f).detach().clone()
        distance = output_f_ @ self.source_f_bank.T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=min(7, distance.shape[1]))
        score_near = self.source_score_bank[idx_near]
        score_near_weight = self.get_weight(score_near)
        score_near_sum_weight = torch.einsum("ijk,ij->ik", score_near, score_near_weight)
        target_predict = torch.nn.functional.softmax(score_near_sum_weight, dim=1)
        return target_predict

    def get_init_banks(self, source, source_index):
        source_index = source_index.long()
        source_f, _ = self.encoder(source)
        source_predict = self.cls_classifier(source_f)
        source_label_feature = torch.nn.functional.softmax(source_predict, dim=1)
        self.source_f_bank[source_index] = torch.nn.functional.normalize(source_f).detach().clone()
        self.source_score_bank[source_index] = source_label_feature.detach().clone()

    def forward(
        self,
        source: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        source_index: Optional[torch.Tensor] = None,
    ):
        if target is None:
            source_f, _ = self.encoder(source)
            source_predict = self.cls_classifier(source_f)
            return source_predict

        if source_index is None:
            raise ValueError("source_index is required when target is provided")

        source_label = torch.zeros((source.shape[0], self.num_of_class), device=source.device)
        source_predict, source_f, target_predict, target_f, _, _, target_label = super().forward(
            source,
            target,
            source_label,
            source_index,
        )
        return source_predict, target_predict, source_f, target_f, target_label
