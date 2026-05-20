import math

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from model.Adversarial import DAANLoss
from model.PCL_TDGCN import Discriminator, PCL
from utils.metric import Metric


class _LabelSmoothingCE(torch.nn.Module):
    def __init__(self, classes: int, epsilon: float = 0.0005):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.epsilon / (input.size(-1) - 1.0)
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.epsilon))
        return (-weight * log_prob).sum(dim=-1).mean()


class _StepwiseLR:
    def __init__(self, optimizer, init_lr: float, gamma: float = 10.0,
                 decay_rate: float = 0.75, max_iter: int = 1000):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.max_iter = max_iter
        self.iter_num = 0

    def step(self):
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** self.decay_rate
        for pg in self.optimizer.param_groups:
            pg.setdefault("lr_mult", 1.0)
            pg["lr"] = lr * pg["lr_mult"]
        self.iter_num += 1


def train(
    model: PCL,
    metric: Metric,
    train_data: np.ndarray,   # (N, 310) float32, band-major
    train_labels: np.ndarray, # (N,) int64
    test_data: np.ndarray,    # (M, 310) float32
    test_labels: np.ndarray,  # (M,) int64
    batch_size: int,
    num_classes: int,
    device: str,
    epochs: int,
    subject_id: int,
    session_id: int,
    learning_rate: float,
    weight_decay: float,
    eval_interval: int,
    early_stop_patience: int,
) -> None:
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_labels = train_labels.astype(np.int64).reshape(-1)
    test_labels = test_labels.astype(np.int64).reshape(-1)

    source_num = train_data.shape[0]
    target_num = test_data.shape[0]

    # --- Datasets and loaders ---
    source_dataset = TensorDataset(
        torch.from_numpy(train_data),
        torch.arange(source_num).long(),
        torch.from_numpy(train_labels),
    )
    target_dataset = TensorDataset(
        torch.from_numpy(test_data),
        torch.arange(target_num).long(),
        torch.from_numpy(test_labels),
    )
    source_loader = DataLoader(source_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(target_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # --- Loss, discriminator, optimizer ---
    criterion = _LabelSmoothingCE(classes=num_classes).to(device)
    discriminator = Discriminator(model.encoder.fc2.out_features).to(device)
    dann_loss = DAANLoss(discriminator, num_class=num_classes).to(device)

    optimizer = torch.optim.RMSprop(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    lr_scheduler = _StepwiseLR(optimizer, init_lr=learning_rate,
                                gamma=10, decay_rate=0.75, max_iter=epochs)

    # --- Initialize memory banks ---
    model.eval()
    with torch.no_grad():
        for src_feat, src_idx, _ in source_loader:
            model.get_init_banks(src_feat.to(device), src_idx.to(device))
        for tgt_feat, tgt_idx, _ in target_loader:
            model.get_init_banks_tgt(tgt_feat.to(device), tgt_idx.to(device))

    best_acc = 0.0
    patience_counter = 0
    num_batches = len(target_loader.dataset) // batch_size

    for epoch in range(epochs):

        # --- Evaluation ---
        if epoch % eval_interval == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for feat, _, label in test_loader:
                    feat, label = feat.to(device), label.to(device)
                    out = model.target_predict(feat)
                    pred = out.argmax(dim=1)
                    correct += pred.eq(label.view_as(pred)).sum().item()
            accuracy = correct / len(test_loader.dataset)

            metric.update(subject_id, session_id, accuracy)

            if accuracy > best_acc:
                best_acc = accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % (eval_interval * 50) == 0 :
                logger.info(
                    "Epoch {}/{} subj {} sess {} | acc={:.4f} best={:.4f}",
                    epoch, epochs, subject_id, session_id, accuracy, best_acc,
                )

            if accuracy >= 1.0:
                logger.info("Perfect accuracy, stopping.")
                break
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                logger.info("Early stop at epoch {} best={:.4f}", epoch, best_acc)
                break

        # --- Training ---
        model.train()
        dann_loss.train()

        src_iter = iter(source_loader)
        tar_iter = iter(target_loader)

        total_loss_sum = 0.0
        for batch_idx in range(num_batches):
            src_feat, src_idx, src_label = next(src_iter)
            tar_feat, tar_idx, _ = next(tar_iter)

            src_feat = src_feat.to(device)
            src_idx = src_idx.to(device)
            src_label = src_label.to(device).view(-1)
            tar_feat = tar_feat.to(device)
            tar_idx = tar_idx.to(device)

            (src_out, src_f, tar_out, tar_f,
             _src_att, _tar_att,
             src_sim, tgt_sim, tgt_cluster_label,
             s2t_pro, t2s_pro, s2s_pro, t2t_pro) = model(
                src_feat, tar_feat, src_label, src_idx, tar_idx, epoch, epochs
            )

            cls_loss = criterion(src_out, src_label)

            src_prob = F.softmax(src_out, dim=1)
            mask = src_prob.max(dim=1).values > 0.7
            if mask.any():
                source_loss = criterion(src_prob[mask], src_label[mask])
            else:
                source_loss = torch.tensor(0.0, device=device)

            target_loss = criterion(tgt_sim, tgt_cluster_label.long())

            global_transfer_loss = dann_loss(
                src_f + 0.005 * torch.randn_like(src_f),
                tar_f + 0.005 * torch.randn_like(tar_f),
                src_prob, F.softmax(tar_out, dim=1),
            )

            boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-epoch / 1000)) - 1)

            s2t_entropy = -(s2t_pro * torch.log(s2t_pro + 1e-10)).sum(dim=1).mean()
            t2s_entropy = -(t2s_pro * torch.log(t2s_pro + 1e-10)).sum(dim=1).mean()
            cross_domain_loss = s2t_entropy + t2s_entropy

            s2s_entropy = -(s2s_pro * torch.log(s2s_pro + 1e-10)).sum(dim=1).mean()
            t2t_entropy = -(t2t_pro * torch.log(t2t_pro + 1e-10)).sum(dim=1).mean()
            in_domain_loss = s2s_entropy + t2t_entropy

            loss = (cls_loss + global_transfer_loss + source_loss
                    + boost_factor * target_loss
                    + 0.2 * (cross_domain_loss + in_domain_loss))

            if torch.isnan(loss):
                logger.warning("NaN loss at epoch {} batch {}, skipping.", epoch, batch_idx)
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item()

        if epoch % (eval_interval * 50) == 0:
            logger.info(
                "Epoch {}/{} | avg_loss={:.4f}",
                epoch, epochs, total_loss_sum / max(num_batches, 1),
            )

        lr_scheduler.step()
