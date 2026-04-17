import numpy as np
import torch
from einops import rearrange
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from model.saber import Saber
from utils.failure_sample import record_failures
from utils.metric import Metric


class SimpleDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)

    def get_data(self):
        return self.tensors


def _to_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = labels.astype(np.int64)
    return np.eye(num_classes, dtype=np.float32)[labels]


def _normalize_for_prpl(train_data: np.ndarray, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror PRPL normalization with separate source/target scalers."""
    src_scaler = MinMaxScaler(feature_range=(-1, 1))
    tgt_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_norm = src_scaler.fit_transform(train_data)
    test_norm = tgt_scaler.fit_transform(test_data)
    return train_norm.astype(np.float32), test_norm.astype(np.float32)


@torch.no_grad()
def _evaluate(model: Saber, dataloader: DataLoader, device: str, return_preds: bool = False):
    model.eval()
    features, labels = dataloader.dataset.get_data()
    if labels.dim() > 1:
        labels_np = np.argmax(labels.numpy(), axis=1)
    else:
        labels_np = labels.numpy()

    y_preds = model.predict(features.to(device)).cpu().numpy()
    acc = np.sum(y_preds == labels_np) / len(labels_np)
    if return_preds:
        return acc * 100.0, y_preds, labels_np
    return acc * 100.0


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
    early_stop_patience: int = 0,
    transfer_loss_weight: float = 1.0,
    cluster_weight: float = 2.0,
    weight_decay: float = 1e-5,
    test_metadata: list | None = None,
    failure_log_path: str | None = None,
):
    logger.info("len of train data: {}, len of test data: {}", len(train_data), len(test_data))

    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)

    train_shape = train_data.shape
    test_shape = test_data.shape
    train_flat = train_data.reshape(train_shape[0], -1)
    test_flat = test_data.reshape(test_shape[0], -1)
    train_flat, test_flat = _normalize_for_prpl(train_flat, test_flat)
    train_data = train_flat.reshape(train_shape)
    test_data = test_flat.reshape(test_shape)

    train_data = rearrange(train_data, "sample chan feature -> sample feature chan")
    test_data = rearrange(test_data, "sample chan feature -> sample feature chan")

    train_label_oh = _to_one_hot(train_labels, num_classes)
    test_labels = test_labels.astype(np.int64)

    source_dataset = SimpleDataset(torch.from_numpy(train_data), torch.from_numpy(train_label_oh))
    target_dataset = SimpleDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    source_loader = DataLoader(
        source_dataset,
        sampler=RandomSampler(source_dataset),
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
    )
    target_loader = DataLoader(
        target_dataset,
        sampler=RandomSampler(target_dataset),
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
    )

    optimizer_params = model.get_parameters() if hasattr(model, "get_parameters") else model.parameters()
    optimizer = torch.optim.RMSprop(
        optimizer_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_acc = 0.0
    stop = 0
    boost_factor = 0.0

    for epoch in range(epochs):
        model.train()

        n_batch = min(len(source_loader), len(target_loader)) - 1
        if n_batch <= 0:
            logger.warning(
                "No valid SABER batch in epoch {}. Check batch_size={} with train/test sizes.",
                epoch + 1,
                batch_size,
            )
            break

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        loss_clf = 0.0
        loss_transfer = 0.0
        loss_cluster = 0.0
        loss_p = 0.0

        for _ in range(n_batch):
            try:
                src_data, src_label = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                src_data, src_label = next(source_iter)

            try:
                tgt_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_data, _ = next(target_iter)

            src_data, src_label = src_data.to(device), src_label.to(device)
            tgt_data = tgt_data.to(device)

            cls_loss, cluster_loss, p_loss, transfer_loss = model(src_data, tgt_data, src_label)
            total_loss = (
                cls_loss
                + transfer_loss_weight * transfer_loss
                + 0.01 * p_loss
                + boost_factor * cluster_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_clf += cls_loss.detach().item()
            loss_transfer += transfer_loss.detach().item()
            loss_cluster += cluster_loss.detach().item()
            loss_p += p_loss.detach().item()

        source_features, source_labels = source_loader.dataset.get_data()
        boost_factor = cluster_weight * ((epoch + 1) / max(1, epochs))
        model.epoch_end_hook(epoch, source_features.to(device), source_labels.to(device))

        target_acc = _evaluate(model, target_loader, device)
        metric.update(subject_id, session_id, target_acc / 100.0)

        if target_acc > best_acc and target_acc > 0.70:
            best_acc = target_acc
            stop = 0
            if test_metadata is not None and failure_log_path is not None:
                _, y_preds, y_true = _evaluate(model, target_loader, device, return_preds=True)
                record_failures(y_true, y_preds, test_metadata, subject_id, session_id, failure_log_path)
        else:
           stop += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            source_acc = _evaluate(model, source_loader, device)

            logger.info(
                "Epoch {}/{} | loss_clf: {:.4f}, loss_transfer: {:.4f}, loss_cluster: {:.4f}, loss_p: {:.4f}, source_acc: {:.4f}, target_acc: {:.4f}, best_acc: {:.4f}",
                epoch + 1,
                epochs,
                loss_clf / n_batch,
                loss_transfer / n_batch,
                loss_cluster / n_batch,
                loss_p / n_batch,
                source_acc,
                target_acc,
                best_acc,
            )

        if early_stop_patience > 0 and stop >= early_stop_patience:
            logger.info("Early stop at epoch {} with best target acc {:.4f}", epoch + 1, best_acc)
            break
