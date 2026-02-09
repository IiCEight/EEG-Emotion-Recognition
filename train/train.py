from loguru import logger
import torch
from torch.utils.data import DataLoader, TensorDataset
from constant import CLI_arguments_enum
from constant.model_map import IS_GRAPH_MODEL, MODEL
from data.seed import SEED_RGNN_ADJACENCY_MATRIX
from train.graph_train import graph_train
from utils.metric import Metric


def train(
    model_name: str,
    split_dataset: list,
    num_subjects: int,
    num_electrodes: int,
    num_features: int,
    num_classes: int,
    device: str,
    task_type: str,
    batch_size: int,
    epochs: int,
    learning_rate: float = 0.001,
):
    edge_adj = None
    if IS_GRAPH_MODEL[model_name]:
        # construct the initial edge adjacency matrix
        edge_adj = torch.Tensor(SEED_RGNN_ADJACENCY_MATRIX)

    if task_type == CLI_arguments_enum.TaskTypeName.SUBJECT_DEPENDENT:
        # train a separate model for each subject
        for subject_id in range(num_subjects):
            logger.info("--- Starting Training for Subject Dependent Model ---")

            # Create DataLoaders
            train_loader = DataLoader(
                dataset=TensorDataset(
                    torch.tensor(split_dataset[0][subject_id]).float(),
                    torch.tensor(split_dataset[1][subject_id]).float(),
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            val_loader = DataLoader(
                dataset=TensorDataset(
                    torch.tensor(split_dataset[2][subject_id]).float(),
                    torch.tensor(split_dataset[3][subject_id]).float(),
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )

            if edge_adj is not None:
                graph_train(
                    model_name,
                    train_loader,
                    val_loader,
                    num_subjects,
                    num_electrodes,
                    num_features,
                    num_classes,
                    device,
                    epochs,
                    learning_rate,
                    subject_id,
                    edge_adj,
                )
            else:
                normal_train(
                    model_name,
                    train_loader,
                    val_loader,
                    num_subjects,
                    num_electrodes,
                    num_features,
                    num_classes,
                    device,
                    epochs,
                    learning_rate,
                )

            # TODO: Test Evaluation for this subject
            # (You might want to return these results to calculate an average later)

    else:
        logger.info("--- Starting Training for Subject Independent Model ---")

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=TensorDataset(
                torch.tensor(split_dataset[0]).float(),
                torch.tensor(split_dataset[1]).float(),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            dataset=TensorDataset(
                torch.tensor(split_dataset[2]).float(),
                torch.tensor(split_dataset[3]).float(),
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        if edge_adj is not None:
            graph_train(
                model_name,
                train_loader,
                val_loader,
                num_subjects,
                num_electrodes,
                num_features,
                num_classes,
                device,
                epochs,
                learning_rate,
                edge_adj=edge_adj,
            )
        else:
            normal_train(
                model_name,
                train_loader,
                val_loader,
                num_subjects,
                num_electrodes,
                num_features,
                num_classes,
                device,
                epochs,
                learning_rate,
            )


def normal_train(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_subjects: int,
    num_electrodes: int,
    num_features: int,
    num_classes: int,
    device: str,
    epochs: int,
    learning_rate: float = 0.001,
    subject_id: int = None,
):
    # logger.info("--- Starting Training ---")

    criterion = torch.nn.CrossEntropyLoss()
    metrics = ["acc"]

    # 1. Initialize model for this subject
    model = MODEL[model_name](num_electrodes, num_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # NOTE: we need initialize a new model for each subject

    # 3. Training Loop
    for epoch in range(epochs):
        # train the model on the training data for one epoch
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()  # Correct placement
            data, labels = batch
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Log progress periodically
        # if (epoch + 1) % 5 == 0:
        evaluate(model, val_loader, device, metrics, criterion)
        avg_loss = epoch_loss / len(train_loader)
        if subject_id is not None:
            logger.info(
                f"Sub {subject_id} | Epoch {epoch + 1}/{epochs} |"
                f" Train Loss: {avg_loss:.4f}"
            )
        else:
            logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f}")

@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion):
    model.eval()
    metric = Metric(metrics)

    # FIXED: Use enumerate or remove idx
    for idx, (samples, labels) in enumerate(data_loader):
        samples = samples.to(device)
        labels = labels.to(device)

        outputs = model(samples)

        # You might not need loss in eval, but if you do:
        loss = criterion(outputs, labels)

        # Update metric (Ensure your Metric class handles raw logits or add softmax here)
        metric.update(torch.argmax(outputs, dim=1), labels, loss.item())

    # Assuming metric.value() returns a string for logging
    logger.info(f"Eval State: {metric.value()}")
