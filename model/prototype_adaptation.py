import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeAdaptation(nn.Module):
    """
    Computes adapted class prototypes by progressively fusing source
    and target domain centroids, following the PAF-CPA paper approach.

    Source prototypes: per-class mean of labeled source features.
    Target prototypes: per-class mean of top-K confident target samples
      (selected by cosine similarity to source prototypes).
    Adapted prototypes: (1 - beta) * source + beta * target
      with beta ramped from 0 -> 1 over training in a stepwise schedule.

    Input features: [batch, feature_dim]
    Output adapted_prototypes: [num_classes, feature_dim]
    """

    def __init__(
        self,
        num_classes: int = 3,
        feature_dim: int = 64,
        max_iter: int = 1000,
        alpha: float = 0.2,
        max_beta: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.max_iter = max_iter
        self.alpha = alpha
        self.max_beta = max_beta  # cap beta to avoid over-reliance on noisy target prototypes

        # Beta schedule: 3 segments, ramping from 0 to max_beta.
        # Matches PAF-CPA's progressive idea but capped at max_beta < 1.0
        # to keep source prototypes dominant.
        self.segment_length = max_iter // 3
        self.ramp_length = self.segment_length // 2

    def _compute_beta(self, epoch: int) -> float:
        """
        Conservative beta schedule: ramps from 0 to max_beta over 3 segments.
        Each segment: first half ramps, second half constant.

        For max_iter=1000, max_beta=0.5:
          Epoch 0-166:   beta 0.00 -> 0.167
          Epoch 167-333: beta 0.167 -> 0.333
          Epoch 334-499: beta 0.333 -> 0.500
          Epoch 500+:    beta = 0.500 (constant)

        This keeps source prototypes as the dominant signal throughout training.
        """
        if self.segment_length <= 0:
            return self.max_beta

        segment = epoch // self.segment_length
        if segment >= 3:
            return self.max_beta

        # Within this segment, ramp from segment*step to (segment+1)*step
        step = self.max_beta / 3.0
        epoch_in_segment = epoch % self.segment_length

        if epoch_in_segment < self.ramp_length and self.ramp_length > 0:
            progress = epoch_in_segment / self.ramp_length
            return segment * step + step * progress
        else:
            return (segment + 1) * step

    def _compute_source_prototypes(
        self, src_feat: torch.Tensor, src_label_oh: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-class centroids from labeled source features.

        Args:
            src_feat: [batch, feature_dim]
            src_label_oh: [batch, num_classes] one-hot labels
        Returns:
            source_prototypes: [num_classes, feature_dim]
        """
        # weights: [num_classes, batch] — each row is the mask for one class
        weights = src_label_oh.t()  # [num_classes, batch]
        counts = weights.sum(dim=1, keepdim=True).clamp(min=1.0)  # [num_classes, 1]
        source_prototypes = weights @ src_feat / counts  # [num_classes, feature_dim]
        return source_prototypes

    def _select_topk_target(
        self, tgt_feat: torch.Tensor, source_prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        For each class, select top-K target samples by cosine similarity
        to the corresponding source prototype.

        Args:
            tgt_feat: [batch, feature_dim]
            source_prototypes: [num_classes, feature_dim]
        Returns:
            masks: [num_classes, batch] binary mask of selected samples
        """
        batch_size = tgt_feat.size(0)

        # Cosine similarity between each target sample and each source prototype
        # tgt_feat: [batch, feature_dim], source_prototypes: [num_classes, feature_dim]
        tgt_norm = F.normalize(tgt_feat, dim=1)  # [batch, feature_dim]
        src_norm = F.normalize(source_prototypes, dim=1)  # [num_classes, feature_dim]
        sim = tgt_norm @ src_norm.t()  # [batch, num_classes]

        k = max(1, int(self.alpha * batch_size))
        k = min(k, batch_size)

        masks = torch.zeros(self.num_classes, batch_size, device=tgt_feat.device)
        for c in range(self.num_classes):
            _, topk_indices = torch.topk(sim[:, c], k)
            masks[c, topk_indices] = 1.0

        return masks

    def _compute_target_prototypes(
        self, tgt_feat: torch.Tensor, masks: torch.Tensor, source_prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute target prototypes as per-class mean of selected samples.
        Falls back to source prototypes if no samples selected for a class.

        Args:
            tgt_feat: [batch, feature_dim]
            masks: [num_classes, batch] binary selection masks
            source_prototypes: [num_classes, feature_dim] fallback
        Returns:
            target_prototypes: [num_classes, feature_dim]
        """
        counts = masks.sum(dim=1, keepdim=True).clamp(min=1.0)  # [num_classes, 1]
        target_prototypes = masks @ tgt_feat / counts  # [num_classes, feature_dim]

        # Fallback: use source prototypes for classes with no selected samples
        has_samples = (masks.sum(dim=1) > 0).float().unsqueeze(1)  # [num_classes, 1]
        target_prototypes = has_samples * target_prototypes + (1 - has_samples) * source_prototypes

        return target_prototypes

    def forward(
        self,
        src_feat: torch.Tensor,
        src_label_oh: torch.Tensor,
        tgt_feat: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """
        Compute adapted prototypes for the current epoch.

        Args:
            src_feat: [batch, feature_dim] source features
            src_label_oh: [batch, num_classes] source one-hot labels
            tgt_feat: [batch, feature_dim] target features
            epoch: current epoch number (0-indexed)
        Returns:
            adapted_prototypes: [num_classes, feature_dim]
        """
        # 1. Source prototypes from labeled data
        source_prototypes = self._compute_source_prototypes(src_feat, src_label_oh)

        # 2. Select top-K confident target samples
        masks = self._select_topk_target(tgt_feat, source_prototypes)

        # 3. Target prototypes from selected samples
        target_prototypes = self._compute_target_prototypes(
            tgt_feat, masks, source_prototypes
        )

        # 4. Progressive fusion with beta schedule
        beta = self._compute_beta(epoch)
        adapted_prototypes = (1 - beta) * source_prototypes + beta * target_prototypes

        return adapted_prototypes
