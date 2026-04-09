import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from model.adann import ADANN
from utils.metric import Metric


class SimpleDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)

    def get_data(self):
        return self.tensors


def _compute_similarity(projected_features, projected_prototypes):
    p_f = F.normalize(projected_features, p=2, dim=1)
    p_mu = F.normalize(projected_prototypes, p=2, dim=1)
    return torch.mm(p_f, p_mu.t())


@torch.no_grad()
def _evaluate(model: ADANN, dataloader: DataLoader, device: str) -> float:
    model.eval()
    features, labels = dataloader.dataset.get_data()
    if labels.dim() > 1:
        labels_np = np.argmax(labels.numpy(), axis=1)
    else:
        labels_np = labels.numpy()
    y_preds = model.predict(features.to(device)).cpu().numpy()
    acc = np.sum(y_preds == labels_np) / len(labels_np)
    return acc * 100.0


@torch.no_grad()
def _init_prototypes(model: ADANN, source_loader: DataLoader, target_loader: DataLoader, device: str, temp: float):
    model.eval()

    all_f_s = []
    all_y_s = []
    for x_s, y_s in source_loader:
        x_s = x_s.to(device)
        f_s = model.feature_extractor(x_s)
        all_f_s.append(f_s)
        all_y_s.append(y_s)

    all_f_s = torch.cat(all_f_s)
    all_y_s = torch.cat(all_y_s).to(device)
    if all_y_s.dim() > 1:
        all_y_s = torch.argmax(all_y_s, dim=1)

    for i in range(model.num_classes):
        mask = all_y_s == i
        if mask.sum() > 0:
            model.mu_s[i] = all_f_s[mask].mean(dim=0)

    all_f_t = []
    for x_t, _ in target_loader:
        x_t = x_t.to(device)
        f_t = model.feature_extractor(x_t)
        all_f_t.append(f_t)
    all_f_t = torch.cat(all_f_t)

    initial_threshold = 0.5 + 0.4
    p_all_f_t = model.projector(all_f_t)
    p_mu_s = model.projector(model.mu_s)

    sim = _compute_similarity(p_all_f_t, p_mu_s)
    probs = F.softmax(sim / temp, dim=1)
    max_probs, pseudo_y_t = probs.max(dim=1)

    mask_reliable = max_probs > initial_threshold
    reliable_f_t = all_f_t[mask_reliable]
    reliable_pseudo_y_t = pseudo_y_t[mask_reliable]

    if reliable_f_t.shape[0] > 0:
        for i in range(model.num_classes):
            mask = reliable_pseudo_y_t == i
            if mask.sum() > 0:
                model.mu_t[i] = reliable_f_t[mask].mean(dim=0)


def train(
    model: ADANN,
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
    w_s: float = 1.0,
    w_t: float = 1.0,
    w_cond: float = 1.0,
    temp: float = 1.0,
    lambda_f_norm: float = 0.01,
    weight_decay: float = 1e-5,
):
    logger.info("len of train data: {}, len of test data: {}", len(train_data), len(test_data))

    train_data = train_data.reshape(train_data.shape[0], -1).astype(np.float32)
    test_data = test_data.reshape(test_data.shape[0], -1).astype(np.float32)

    train_labels = train_labels.astype(np.int64)
    test_labels = test_labels.astype(np.int64)

    source_dataset = SimpleDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
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

    optimizer = torch.optim.RMSprop(
        model.get_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    ce_loss = nn.CrossEntropyLoss()

    _init_prototypes(model, source_loader, target_loader, device, temp)

    best_acc = 0.0
    stop = 0

    for epoch in range(epochs):
        model.train()

        n_batch = min(len(source_loader), len(target_loader)) - 1
        if n_batch <= 0:
            logger.warning(
                "No valid ADANN batch in epoch {}. Check batch_size={} with train/test sizes.",
                epoch + 1,
                batch_size,
            )
            break

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        threshold = 0.5 + 0.4 * (1.0 - epoch / max(1, epochs))

        total_loss = 0.0
        total_dann = 0.0
        total_s = 0.0
        total_t = 0.0
        total_cond = 0.0

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

            if src_label.dim() > 1:
                src_label = torch.argmax(src_label, dim=1)

            out = model(src_data, tgt_data)
            loss_dann = out["loss_dann"]
            f_s, p_s = out["f_s"], out["p_s"]
            f_t, p_t = out["f_t"], out["p_t"]

            p_mu_s = model.projector(model.mu_s)
            p_mu_t = model.projector(model.mu_t)

            sim_s = _compute_similarity(p_s, p_mu_s)
            loss_s = ce_loss(sim_s / temp, src_label)

            with torch.no_grad():
                sim_t_infer = _compute_similarity(p_t, p_mu_s)
                probs_t = F.softmax(sim_t_infer / temp, dim=1)
                max_probs, pseudo_y_t = probs_t.max(dim=1)
                mask_reliable = max_probs > threshold

            loss_t = torch.tensor(0.0, device=device)
            if mask_reliable.sum() > 0:
                sim_t_reliable = _compute_similarity(p_t[mask_reliable], p_mu_t)
                loss_t = ce_loss(sim_t_reliable / temp, pseudo_y_t[mask_reliable])

            p_mu_s_norm = F.normalize(p_mu_s, p=2, dim=1)
            p_mu_t_norm = F.normalize(p_mu_t, p=2, dim=1)
            sim_cond = torch.mm(p_mu_s_norm, p_mu_t_norm.t())
            labels_cond = torch.arange(model.num_classes, device=device)
            loss_cond = ce_loss(sim_cond / temp, labels_cond)

            f_norm_loss = torch.norm(model.projector.weight, p="fro")

            loss = (
                w_s * loss_s
                + w_t * loss_t
                + w_cond * loss_cond
                + loss_dann
                + lambda_f_norm * f_norm_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_s_dist = sim_s.argmax(dim=1)
                pred_t_on_s = pseudo_y_t

                sim_t_on_t = _compute_similarity(p_t, p_mu_t)
                pred_t_on_t = sim_t_on_t.argmax(dim=1)

                agreement = (pred_t_on_s == pred_t_on_t).float().mean().item()
                gamma = 1.0 / (18.82 * agreement - 20.0) + 1.05

                hat_mu_s = model.mu_s.clone()
                for i in range(model.num_classes):
                    mask = pred_s_dist == i
                    if mask.sum() > 0:
                        hat_mu_s[i] = f_s[mask].mean(dim=0)
                model.mu_s.data = (1.0 - gamma) * hat_mu_s + gamma * model.mu_s.data

                sim_s_on_t = _compute_similarity(p_s, p_mu_t)
                pred_s_on_t = sim_s_on_t.argmax(dim=1)
                mask_aligned_s = pred_s_on_t == src_label

                features_for_update = []
                labels_for_update = []

                if mask_reliable.sum() > 0:
                    features_for_update.append(f_t[mask_reliable])
                    labels_for_update.append(pseudo_y_t[mask_reliable])

                if mask_aligned_s.sum() > 0:
                    features_for_update.append(f_s[mask_aligned_s])
                    labels_for_update.append(src_label[mask_aligned_s])

                if len(features_for_update) > 0:
                    combined_f = torch.cat(features_for_update)
                    combined_y = torch.cat(labels_for_update)

                    hat_mu_t = model.mu_t.clone()
                    for i in range(model.num_classes):
                        mask = combined_y == i
                        if mask.sum() > 0:
                            hat_mu_t[i] = combined_f[mask].mean(dim=0)
                    model.mu_t.data = (1.0 - gamma) * hat_mu_t + gamma * model.mu_t.data

            total_loss += loss.item()
            total_dann += loss_dann.item()
            total_s += loss_s.item()
            total_t += loss_t.item()
            total_cond += loss_cond.item()

        source_acc = _evaluate(model, source_loader, device)
        target_acc = _evaluate(model, target_loader, device)
        metric.update(subject_id, session_id, target_acc / 100.0)

        if target_acc > best_acc:
            best_acc = target_acc
            stop = 0
        else:
            stop += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch {}/{} | total_loss: {:.4f}, loss_dann: {:.4f}, loss_s: {:.4f}, loss_t: {:.4f}, loss_cond: {:.4f}, source_acc: {:.4f}, target_acc: {:.4f}, best_acc: {:.4f}",
                epoch + 1,
                epochs,
                total_loss / n_batch,
                total_dann / n_batch,
                total_s / n_batch,
                total_t / n_batch,
                total_cond / n_batch,
                source_acc,
                target_acc,
                best_acc,
            )

        if early_stop_patience > 0 and stop >= early_stop_patience:
            logger.info("Early stop at epoch {} with best target acc {:.4f}", epoch + 1, best_acc)
            break