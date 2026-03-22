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

from model.NSAL_DGAT import DAANLoss, Discriminator
from constant import CLI_arguments_enum
from utils.metric import Metric


def train(
    model: nn.Module,
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

    hidden_2 = 64
    domain_discriminator = Discriminator(hidden_2).to(device)
    logger.info("num_classes {}", num_classes)
    dann_loss = DAANLoss(domain_discriminator, num_class=num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(domain_discriminator.parameters()),
        lr=learning_rate,
        weight_decay=0.001,
    )

    #  TODO.
    decay_math = lambda epoch: 1.0 / (1.0 + 10 * (epoch / max(1, epochs))) ** 0.75
    scheduler = LambdaLR(optimizer, lr_lambda=decay_math)

    lr_scheduler = StepwiseLR_GRL(optimizer, init_lr=learning_rate, gamma=10, decay_rate=0.75, max_iter=epochs)


    model.train()

    # getInit(train_loader, model, device)

    test_data = torch.tensor(test_data).float()
    test_labels = torch.tensor(test_labels).long()

    for epoch in range(1, epochs + 1):
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
            # data = rearrange(data, 'b chan feature -> b feature chan', chan= 62, feature = 5)
            # target_data = rearrange(target_data, 'b chan feature -> b feature chan', chan= 62, feature = 5)

            output, feature, target_output, target_feature, _, _, target_labels = model(
                data,
                target_data,
                labels,
                index,
            )
            source_loss = criterion(output, labels)
            # target_labels = torch.argmax(target_labels, dim=1)
            # target_loss = criterion(target_output, target_labels)
            global_transfer_loss = dann_loss(
                feature + 0.005 * torch.randn((feature.shape[0], (hidden_2))).to(device),
                target_feature + 0.005 * torch.randn((target_feature.shape[0], (hidden_2))).to(device),
                output, target_output)
            # model.eval()
            boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-1 * (epoch-1) / 1000)) - 1)
            # loss = source_loss + global_transfer_loss + boost_factor * target_loss
            
            # delete clustering component
            loss = source_loss + global_transfer_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        # lr_scheduler.step()
        evaluate(model, metric, test_data, test_labels, device, subject_id, session_id)
        avg_loss = epoch_loss / len(train_loader)

        if epoch % 5 == 0:
            logger.info("Epoch {}/{} | Train Loss: {:.4f}", epoch, epochs, avg_loss)
            logger.info("Current lr = {:.6f}", scheduler.get_last_lr()[0])
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
    
    outputs = model.target_predict(data)
    predictions = torch.argmax(outputs, dim=1)

    correct_in_batch = (predictions == labels).sum().item()
    acc = correct_in_batch / labels.size(0)
    metric.update(subject_id, session_id, acc)


def pytorch_safe_cycle(iterable):
    while True:
        for x in iterable:
            yield x
