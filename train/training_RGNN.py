
import itertools
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from einops import rearrange
from loguru import logger
import numpy as np
import torch

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler, RandomSampler, SequentialSampler, TensorDataset
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
    learning_rate: float = 0.001,
):
    
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

    # generate domain labels for source and target data
    source_domain_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    target_domain_labels = torch.ones(batch_size, dtype=torch.long, device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Define the DANN decay math
    # PyTorch automatically multiplies this result by your initial_lr
    decay_math = lambda epoch: 1.0 / (1.0 + 10 * (epoch / epochs)) ** 0.75
    # Attach the built-in scheduler to your optimizer
    scheduler = LambdaLR(optimizer, lr_lambda=decay_math)

    # 1. Initialize the Cosine Scheduler
    # T_max is the number of steps until the LR hits the minimum. 
    # eta_min is the lowest the LR will go (prevents it from hitting absolute zero).
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=epochs,  # Set this to your total epochs
    #     eta_min=1e-4
    # )

    for epoch in range(epochs):

        # train the model on the training data for one epoch
        model.train()
        epoch_loss = 0.0

        for data, labels in train_loader:
            optimizer.zero_grad()

            data, labels = data.to(device), labels.to(device)
            # hidden the taget lables.
            target_data, _ = next(target_loader_inf_iter)
            target_data = target_data.to(device)

            output, domain, target_output, target_domain = model(data, target_data)

            # NOTE: If the label is not one-hot encoded, the type of labels 
            # should be long for CrossEntropyLoss
            loss = criterion(output, labels.long())

            # generate domain labels for source and target data
            # source domain label: 0, target domain label: 1
            loss += 0.5 * criterion(domain, target_domain_labels) + criterion(target_domain, source_domain_labels)

            loss.backward()

            #  Add a gradient clipping step to prevent exploding gradients
            # clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()  # Update the learning rate based on the decay math

        # Log progress periodically
        evaluate(model, metric, test_data, test_labels, device, criterion, subject_id, session_id)
        avg_loss = epoch_loss / len(train_loader)

        show_log_per_epoch = 10 if task_type == CLI_arguments_enum.TaskTypeName.SUBJECT_DEPENDENT else 5
        if (epoch + 1) % show_log_per_epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")
            logger.info("Current lr = {:<.6f}", scheduler.get_last_lr()[0])

@torch.no_grad()
def evaluate(model, metric:Metric, data, labels, device, criterion, subject_id, session_id):
    """
        The shaple of data is (session, sample, electrode, feature), 
            and the shape of labels is (session, sample)
    """
    model.eval()

    data = data.to(device)
    labels = labels.to(device)
    
    outputs, _ = model(data)
    
    # You might not need loss in eval, but if you do:
    # loss = criterion(outputs, labels.long())

    # shape of outputs: [batch_size, num_classes]
    _, predictions = torch.max(outputs, dim=1)


    # 2. Compare predictions to actual labels and count the matches
    correct_in_batch = (predictions == labels).sum().item()

    acc = correct_in_batch / labels.size(0)


    # logger.info("Session {}: Accuracy: {:.4f}, Loss: {:.4f}", session_id, acc, loss.item())
    
    metric.update(subject_id, session_id, acc)
    

# Put this helper function at the top of your file
def pytorch_safe_cycle(iterable):
    """Infinitely loops a PyTorch DataLoader, safely reshuffling every epoch."""
    while True:
        for x in iterable:
            yield x