from typing import Optional

from einops import rearrange
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from model.saber import Saber
from utils.metric import Metric


def _to_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = labels.astype(np.int64)
    return np.eye(num_classes, dtype=np.float32)[labels]


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
    cluster_weight: float = 2.0,
):

    logger.info('len of train data: {}, len of test data: {}', len(train_data), len(test_data))

    train_data = rearrange(train_data, 'sample chan feature -> sample feature chan', chan=62, feature=5)
    test_data = rearrange(test_data, 'sample chan feature -> sample feature chan', chan=62, feature=5)

    train_label_oh = _to_one_hot(train_labels, num_classes)
    test_labels = test_labels.astype(np.int64)

    dataset_train = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label_oh))
    dataset_test = TensorDataset(torch.Tensor(test_data), torch.LongTensor(test_labels))

    sampler_train = RandomSampler(dataset_train)
    sampler_test = SequentialSampler(dataset_test)

    train_loader = DataLoader(
        dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4, drop_last=True
    )
    test_loader = DataLoader(
        dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4, drop_last=True
    )

    source_loader_inf_iter = pytorch_safe_cycle(train_loader)
    target_loader_inf_iter = pytorch_safe_cycle(test_loader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.001,
    )

    decay_math = lambda epoch: 1.0 / (1.0 + 10 * (epoch / max(1, epochs))) ** 0.75
    scheduler = LambdaLR(optimizer, lr_lambda=decay_math)

    train_data_tensor = torch.tensor(train_data).float()
    train_labels_oh_tensor = torch.tensor(train_label_oh).float()
    test_data_tensor = torch.tensor(test_data).float()
    test_labels_tensor = torch.tensor(test_labels).long()

    patience = early_stop_patience
    best_acc = 0.0
    epochs_without_improvement = 0
    boost_factor = 0.0

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        clf_loss_total = 0.0
        cluster_loss_total = 0.0
        p_loss_total = 0.0

        num_batches = min(len(train_loader), len(test_loader))

        for _ in range(num_batches):
            optimizer.zero_grad()

            data, source_label_oh = next(source_loader_inf_iter)
            target_data, _ = next(target_loader_inf_iter)

            data = data.to(device)
            source_label_oh = source_label_oh.to(device)
            target_data = target_data.to(device)

            clf_loss, cluster_loss, p_loss = model(data, target_data, source_label_oh)

            loss = clf_loss + 0.01 * p_loss + boost_factor * cluster_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            clf_loss_total += clf_loss.item()
            cluster_loss_total += cluster_loss.item()
            p_loss_total += p_loss.item()

        scheduler.step()

        boost_factor = cluster_weight * (epoch / max(1, epochs))
        model.epoch_end_hook(
            epoch - 1,
            train_data_tensor.to(device),
            train_labels_oh_tensor.to(device),
        )

        source_acc = evaluate_all(
            model,
            train_data_tensor,
            torch.argmax(train_labels_oh_tensor, dim=1),
            device,
        )
        target_acc = evaluate_all(
            model,
            test_data_tensor,
            test_labels_tensor,
            device,
        )
        metric.update(subject_id, session_id, target_acc)

        current_acc = metric.accuracy[subject_id, session_id]

        if current_acc >= 1.0 - 1e-6:
            logger.info('Early stop at epoch {} — perfect accuracy reached ({:.4f})', epoch, current_acc)
            break

        if current_acc > best_acc + 1e-6:
            best_acc = current_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            logger.info(
                'Early stop at epoch {} — no improvement for {} epochs (best={:.4f})',
                epoch,
                patience,
                best_acc,
            )
            break

        total_loss /= num_batches
        clf_loss_avg = clf_loss_total / num_batches
        cluster_loss_avg = cluster_loss_total / num_batches
        p_loss_avg = p_loss_total / num_batches

        if epoch % 5 == 0:
            logger.info(
                'Epoch {}/{} | Total Loss: {:.4f}, Clf: {:.4f}, Cluster: {:.4f}, P: {:.4f}',
                epoch,
                epochs,
                total_loss,
                clf_loss_avg,
                cluster_loss_avg,
                p_loss_avg,
            )
            logger.info('cluster loss weight: {:.4f}', boost_factor)
            logger.info('Current lr = {:.6f}', scheduler.get_last_lr()[0])
            logger.info('Source Accuracy: {:.4f}, Target Accuracy: {:.4f}', source_acc, target_acc)


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
    data: torch.Tensor,
    labels: torch.Tensor,
    device: str,
):
    model.eval()

    data = data.to(device)
    labels = labels.to(device)

    outputs = model.predict(data)
    predictions = outputs.long() if outputs.ndim == 1 else torch.argmax(outputs, dim=1)

    correct_in_batch = (predictions == labels).sum().item()
    acc = correct_in_batch / labels.size(0)
    return acc


def pytorch_safe_cycle(iterable):
    while True:
        for x in iterable:
            yield x
