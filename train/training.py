import math
from typing import Optional
from einops import rearrange
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from constant import CLI_arguments_enum
from model.saber import Saber
from utils.metric import Metric

import torch.nn.functional as F


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
    early_stop_patience: int = 15,
):

    logger.info("len of train data: {}, len of test data: {}", len(train_data), len(test_data))

    train_data = rearrange(train_data, 'sample chan feature -> sample feature chan', chan= 62, feature = 5)
    test_data =  rearrange(test_data, 'sample chan feature -> sample feature chan', chan= 62, feature = 5)

    dataset_train =TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels))
    dataset_test = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_labels))

    sampler_train = RandomSampler(dataset_train)
    sampler_test = SequentialSampler(dataset_test)

    train_loader = DataLoader(
        dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4, drop_last=True
    )

    test_loader = DataLoader(
        dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4, drop_last=True
    )

    target_loader_inf_iter = pytorch_safe_cycle(test_loader)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.001,
    )

    decay_math = lambda epoch: 1.0 / (1.0 + 10 * (epoch / max(1, epochs))) ** 0.75
    scheduler = LambdaLR(optimizer, lr_lambda=decay_math)

    test_data = torch.tensor(test_data).float()
    test_labels = torch.tensor(test_labels).long()

    # Domain labels for source and target
    source_domain_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    target_domain_labels = torch.ones(batch_size, dtype=torch.long, device=device)

    # Early stopping state
    patience = early_stop_patience   # 0 = disabled
    best_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        source_loss_total = 0.0
        domain_loss_total = 0.0
        correct = 0
        total_samples = 0

        for data, labels in train_loader:
            optimizer.zero_grad()

            data = data.to(device)
            labels = labels.long().to(device)

            target_data, _ = next(target_loader_inf_iter)
            target_data = target_data.to(device)

            (output, domain_output_source, domain_output_target,
             fused_feat) = model(data, target_data)

            # --- Main losses ---
            source_loss = criterion(output, labels)
            domain_loss = 0.5 * criterion(domain_output_source, source_domain_labels)
            domain_loss += criterion(domain_output_target, target_domain_labels)

            loss = source_loss + domain_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            source_loss_total += source_loss.item()
            domain_loss_total += domain_loss.item()

            # Track train accuracy
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()
        evaluate_all(model, metric, test_data, test_labels, device, subject_id, session_id)

        # --- Early stopping check ---
        current_acc = metric.accuracy[subject_id, session_id]

        if current_acc >= 1.0 - 1e-6:
            logger.info("Early stop at epoch {} — perfect accuracy reached ({:.4f})",
                        epoch, current_acc)
            break

        if current_acc > best_acc + 1e-6:
            best_acc = current_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            logger.info("Early stop at epoch {} — no improvement for {} epochs (best={:.4f})",
                        epoch, patience, best_acc)
            break

        n_batches = len(train_loader)
        total_loss /= n_batches
        source_loss_avg = source_loss_total / n_batches
        domain_loss_avg = domain_loss_total / n_batches
        train_acc = correct / total_samples if total_samples > 0 else 0.0

        if epoch % 5 == 0:
            logger.info("Epoch {}/{} | Total Loss: {:.4f}, Source Loss: {:.4f}, Domain "
                        + "Loss: {:.4f}",
                        epoch, epochs, total_loss, source_loss_avg, domain_loss_avg)
            logger.info("Current lr = {:.6f}", scheduler.get_last_lr()[0])
            logger.info("Train Accuracy: {:.4f}", train_acc)


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


@torch.no_grad()
def evaluate_all(
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

    outputs = model.predict(data)
    predictions = torch.argmax(outputs, dim=1)

    correct_in_batch = (predictions == labels).sum().item()
    acc = correct_in_batch / labels.size(0)
    metric.update(subject_id, session_id, acc)


def pytorch_safe_cycle(iterable):
    while True:
        for x in iterable:
            yield x
