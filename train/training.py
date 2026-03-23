import math
from typing import Optional
from einops import rearrange
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from model.NSAL_DGAT import DAANLoss, Discriminator
from constant import CLI_arguments_enum
from model.saber import Saber
from utils.metric import Metric


# ---------------------------------------------------------------------------
# Polarity label helpers
# ---------------------------------------------------------------------------

def make_polarity_labels(labels: torch.Tensor):
    """
    Convert 3-class emotion labels into two binary polarity targets.

    Original label mapping (SEED):
        0 = Negative, 1 = Neutral, 2 = Positive

    Returns:
        polarity_a  – binary target for branch A  (1 = Positive,  0 = otherwise)
        polarity_b  – binary target for branch B  (1 = Negative,  0 = otherwise)
    """
    polarity_a = (labels == 2).long()   # positive-vs-rest
    polarity_b = (labels == 0).long()   # negative-vs-rest
    return polarity_a, polarity_b


def orthogonality_loss(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """
    Encourage the two branch representations to be decorrelated.

    L_ortho = mean( (norm_a · norm_b)^2 )

    Args:
        feat_a: [B, D]  branch A projected features
        feat_b: [B, D]  branch B projected features
    Returns:
        scalar loss
    """
    norm_a = F.normalize(feat_a, p=2, dim=1)
    norm_b = F.normalize(feat_b, p=2, dim=1)
    cosine = (norm_a * norm_b).sum(dim=1)         # [B]
    return (cosine ** 2).mean()


import torch.nn.functional as F   # used by orthogonality_loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    Pulls features of the same class together and pushes features of
    different classes apart in the embedding space.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, D]  L2-normalized feature vectors
            labels:   [B]     class labels
        Returns:
            scalar loss
        """
        features = F.normalize(features, dim=1)
        batch_size = features.size(0)

        # Pairwise cosine similarity / temperature
        similarity = features @ features.T / self.temperature   # [B, B]

        # Positive mask: same class, excluding self
        labels_col = labels.unsqueeze(0)       # [1, B]
        labels_row = labels.unsqueeze(1)       # [B, 1]
        positive_mask = (labels_row == labels_col).float()     # [B, B]
        self_mask = 1.0 - torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * self_mask              # exclude diagonal

        # Numerically stable log-softmax over non-self entries
        # Subtract max for numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Average log-prob over positive pairs
        num_positives = positive_mask.sum(dim=1)                # [B]
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

        # Only count samples that have at least one positive pair
        valid = (num_positives > 0).float()
        loss = -(mean_log_prob * valid).sum() / (valid.sum() + 1e-8)
        return loss


def train(
    model: Saber,
    metric: Metric,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
    num_classes: int,
    device: str,
    epochs: int,
    task_type: str,
    subject_id: int,
    session_id: int,
    learning_rate: float,
):

    logger.info("len of train data: {}, len of test data: {}", len(train_data), len(test_data))

    train_data = rearrange(train_data, 'sample chan feature -> sample feature chan', chan= 62, feature = 5)
    test_data =  rearrange(test_data, 'sample chan feature -> sample feature chan', chan= 62, feature = 5)

    dataset_train =TensorDataset(torch.Tensor(train_data),torch.arange(len(train_data)).long(), torch.Tensor(train_labels))
    dataset_test = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_labels))

    sampler_train = RandomSampler(dataset_train)
    sampler_test = SequentialSampler(dataset_test)

    train_loader = DataLoader(
        dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4, drop_last=True
    )

    test_loader = DataLoader(
        dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4,drop_last=True
    )

    target_loader_inf_iter = pytorch_safe_cycle(test_loader)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = torch.nn.CrossEntropyLoss()   # no smoothing for domain
    criterion_aux = torch.nn.CrossEntropyLoss()       # for auxiliary polarity heads
    supcon_criterion = SupConLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.001,
    )

    # Smooth cosine annealing — no restarts, decays over full training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    # Auxiliary loss weights
    lambda_aux_max = 0.3
    lambda_ortho = 0.05
    lambda_con = 0.05         # supervised contrastive loss weight
    warmup_epochs = 15        # linearly ramp auxiliary losses over this many epochs

    test_data = torch.tensor(test_data).float()
    test_labels = torch.tensor(test_labels).long()

    # generate domain labels for source and target data
    source_domain_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    target_domain_labels = torch.ones(batch_size, dtype=torch.long, device=device)

    for epoch in range(1, epochs + 1):
        # Warmup: linearly increase auxiliary weight from 0 → lambda_aux_max
        warmup_ratio = min(1.0, epoch / max(1, warmup_epochs))
        lambda_aux = lambda_aux_max * warmup_ratio
        model.train()
        epoch_loss = 0.0

        iter = 0
        for data, index, labels in train_loader:
            iter += 1
            # No prefix means it's the source domain.
            # TODO
            optimizer.zero_grad()

            data = data.to(device)
            labels = labels.long().to(device)
            index = index.long().to(device)

            target_data, _ = next(target_loader_inf_iter)
            target_data = target_data.to(device)

            # ---- Mixup augmentation (α=0.4) ----
            mixup_alpha = 0.4
            lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
            perm = torch.randperm(data.size(0), device=device)
            data_mixed = lam * data + (1 - lam) * data[perm]
            labels_perm = labels[perm]

            (output, domain_output_source, domain_output_target,
             aux_a_logits, aux_b_logits,
             branch_a_feat, branch_b_feat,
             fused_feat) = model(data_mixed, target_data)

            # --- Main losses (mixup-aware) ---
            source_loss = lam * criterion(output, labels) + (1 - lam) * criterion(output, labels_perm)
            domain_loss = 0.5 * criterion_domain(domain_output_source, source_domain_labels)
            domain_loss += criterion_domain(domain_output_target, target_domain_labels)

            # --- Auxiliary polarity specialization losses (mixup-aware) ---
            polarity_a, polarity_b = make_polarity_labels(labels)
            polarity_a_perm, polarity_b_perm = make_polarity_labels(labels_perm)
            aux_loss_a = lam * criterion_aux(aux_a_logits, polarity_a) + (1 - lam) * criterion_aux(aux_a_logits, polarity_a_perm)
            aux_loss_b = lam * criterion_aux(aux_b_logits, polarity_b) + (1 - lam) * criterion_aux(aux_b_logits, polarity_b_perm)

            # --- Orthogonality regularization ---
            ortho = orthogonality_loss(branch_a_feat, branch_b_feat)

            # --- Supervised contrastive loss (skip when heavily mixed) ---
            # SupCon assumes features cleanly belong to one class;
            # Mixup-interpolated features violate this → can cause NaN
            if lam > 0.9:
                con_loss = supcon_criterion(fused_feat, labels)
            else:
                con_loss = torch.tensor(0.0, device=device)

            loss = (source_loss
                    + domain_loss
                    + lambda_aux * (aux_loss_a + aux_loss_b)
                    + lambda_ortho * ortho
                    + lambda_con * con_loss)

            # NaN safety guard — must check BEFORE backward
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        # lr_scheduler.step()
        evaluate(model, metric, test_data, test_labels, device, subject_id, session_id)
        avg_loss = epoch_loss / len(train_loader)

        if epoch % 5 == 0:
            logger.info("Epoch {}/{} | Train Loss: {:.4f}", epoch, epochs, avg_loss)
            logger.info("Current lr = {:.6f}", optimizer.param_groups[0]['lr'])
            # logger.info("Current lr = {:.6f}", lr_scheduler.get_lr())




class StepwiseLR_GRL:
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75, max_iter: Optional[float] = 1000):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


def getInit(train_loader, model, device):
    model.eval()
    for _, (tran_input, tran_indx, _ ) in enumerate(train_loader):
        tran_input, tran_indx = tran_input.to(device), tran_indx.to(device)
        model.get_init_banks(tran_input, tran_indx)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    metric: Metric,
    data: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    subject_id: int,
    session_id: int,
):
    model.eval()

    data = data.to(device)
    labels = labels.to(device)
    # data = rearrange(data, 'b chan feature -> b feature chan', chan= 62, feature = 5)
    
    outputs = model.predict(data)
    predictions = torch.argmax(outputs, dim=1)

    correct_in_batch = (predictions == labels).sum().item()
    acc = correct_in_batch / labels.size(0)
    metric.update(subject_id, session_id, acc)


def pytorch_safe_cycle(iterable):
    while True:
        for x in iterable:
            yield x
