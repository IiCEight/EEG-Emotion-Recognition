import math
from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from constant import CLI_arguments_enum
from model.NSAL_DGAT import DAANLoss, Discriminator, NSALDGAT
from utils.metric import Metric


def train(
    model: NSALDGAT,
    metric: Metric,
    train_loader: DataLoader,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
    device: str,
    epochs: int,
    task_type: str,
    subject_id: int,
    session_id: int,
    learning_rate: float = 1e-3,
):
    criterion = torch.nn.CrossEntropyLoss()

    source_loader = DataLoader(
        dataset=train_loader.dataset,
        sampler=RandomSampler(train_loader.dataset),
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
    )

    if task_type == CLI_arguments_enum.TaskTypeName.SUBJECT_INDEPENDENT:
        target_dataset = TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_labels).long())
        target_loader = DataLoader(
            dataset=target_dataset,
            sampler=SequentialSampler(target_dataset),
            batch_size=batch_size,
            num_workers=4,
            drop_last=True,
        )
        target_loader_iter = enumerate(target_loader)
    else:
        target_loader = None
        target_loader_iter = None

    domain_discriminator = Discriminator(hidden_1=model.source_f_bank.shape[1]).to(device)
    dann_loss = DAANLoss(domain_discriminator, max_iter=max(1, epochs * len(source_loader))).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(domain_discriminator.parameters()),
        lr=learning_rate,
        weight_decay=0.001,
    )

    decay_math = lambda epoch: 1.0 / (1.0 + 10 * (epoch / max(1, epochs))) ** 0.75
    scheduler = LambdaLR(optimizer, lr_lambda=decay_math)

    model.train()
    model.reset_source_bank(len(source_loader.dataset), device=torch.device(device))
    _initialize_source_bank(model, source_loader, device)

    test_data = torch.tensor(test_data).float()
    test_labels = torch.tensor(test_labels).long()
    iteration = math.ceil(len(source_loader.dataset) / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        source_loader_iter = enumerate(source_loader)

        for _ in range(iteration):
            try:
                _, batch = next(source_loader_iter)
            except Exception:
                source_loader_iter = enumerate(source_loader)
                _, batch = next(source_loader_iter)

            if len(batch) == 3:
                data, source_index, labels = batch
            else:
                data, labels = batch
                source_index = torch.arange(data.size(0), dtype=torch.long)

            data = data.to(device)
            labels = labels.long().to(device)
            source_index = source_index.long().to(device)

            optimizer.zero_grad()

            if task_type == CLI_arguments_enum.TaskTypeName.SUBJECT_INDEPENDENT:
                try:
                    _, (target_data, _) = next(target_loader_iter)
                except Exception:
                    target_loader_iter = enumerate(target_loader)
                    _, (target_data, _) = next(target_loader_iter)
                target_data = target_data.to(device)

                src_logits, tgt_logits, src_feat, tgt_feat, pseudo_target = model(
                    source=data,
                    target=target_data,
                    source_index=source_index,
                )

                cls_loss = criterion(src_logits, labels)
                pseudo_labels = torch.argmax(pseudo_target.detach(), dim=1)
                target_loss = criterion(tgt_logits, pseudo_labels)
                transfer_loss = dann_loss(
                    src_feat + 0.005 * torch.randn_like(src_feat),
                    tgt_feat + 0.005 * torch.randn_like(tgt_feat),
                    src_logits,
                    tgt_logits,
                )

                boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-1.0 * epoch / 1000.0)) - 1.0)
                loss = cls_loss + transfer_loss + boost_factor * target_loss
            else:
                src_logits = model(source=data)
                loss = criterion(src_logits, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        evaluate(model, metric, test_data, test_labels, device, subject_id, session_id)

        if epoch % 5 == 0:
            avg_loss = epoch_loss / max(1, iteration)
            logger.info("Epoch {}/{} | Train Loss: {:.4f}", epoch, epochs, avg_loss)
            logger.info("Current lr = {:.6f}", scheduler.get_last_lr()[0])


@torch.no_grad()
def _initialize_source_bank(model: NSALDGAT, train_loader: DataLoader, device: str) -> None:
    model.eval()
    for batch in train_loader:
        if len(batch) == 3:
            data, source_index, _ = batch
        else:
            data, _ = batch
            source_index = torch.arange(data.size(0), dtype=torch.long)

        data = data.to(device)
        source_index = source_index.long().to(device)

        if hasattr(model, "get_init_banks"):
            model.get_init_banks(data, source_index)
            continue

        source_feat, _ = model.encoder(data)
        if hasattr(model, "classifier"):
            source_logits = model.classifier(source_feat)
        else:
            source_logits = model.cls_classifier(source_feat)
        source_probs = torch.softmax(source_logits, dim=1)

        model.source_f_bank[source_index] = torch.nn.functional.normalize(source_feat, dim=1).detach()
        model.source_score_bank[source_index] = source_probs.detach()


@torch.no_grad()
def evaluate(
    model: NSALDGAT,
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

    outputs = model(source=data)
    predictions = torch.argmax(outputs, dim=1)

    correct_in_batch = (predictions == labels).sum().item()
    acc = correct_in_batch / labels.size(0)

    metric.update(subject_id, session_id, acc)


def pytorch_safe_cycle(iterable):
    while True:
        for x in iterable:
            yield x
