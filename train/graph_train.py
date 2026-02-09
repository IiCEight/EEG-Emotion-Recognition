from loguru import logger
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from constant import CLI_arguments_enum
from constant.model_map import MODEL
from utils.metric import Metric



def graph_train(
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
    edge_adj = None,
):
    # logger.info("--- Starting Training ---")
    
    criterion = torch.nn.CrossEntropyLoss()
    metrics = ["acc"]

    # 1. Initialize model for this subject
    model = MODEL[model_name](num_electrodes, num_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # NOTE: we need initialize a new model for each subject


    # transform the edge_adj to edge_index(the torch_geometric requested structure)
    edge_index = edge_adj.to_sparse()._indices()

    # 3. Training Loop
    for epoch in range(epochs):

        # train the model on the training data for one epoch
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            edge_index = edge_index.to(device)

            # Use torch_geometric's Data structure to feed the graph data into the model
            input = Data(x=data, edge_index=edge_index)

            outputs = model(input)
            # NOTE: If the label is not one-hot encoded, the type of labels 
            # should be long for CrossEntropyLoss
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Log progress periodically
        # if (epoch + 1) % 5 == 0:
        evaluate(model, val_loader, device, metrics, criterion, edge_index)
        avg_loss = epoch_loss / len(train_loader)
        if subject_id is not None:
            logger.info(f"Sub {subject_id} | Epoch {epoch+1}/{epochs} |"
                        f" Train Loss: {avg_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")



@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, edge_index):
    model.eval()
    metric = Metric(metrics)
    
    # FIXED: Use enumerate or remove idx
    for idx, (samples, labels) in enumerate(data_loader):
        samples = samples.to(device)
        labels = labels.to(device)

        # Use torch_geometric's Data structure to feed the graph data into the model
        input = Data(x=samples, edge_index=edge_index.to(device))
        
        outputs = model(input)
        
        # You might not need loss in eval, but if you do:
        loss = criterion(outputs, labels.long())
        
        # Update metric (Ensure your Metric class handles raw logits or add softmax here)
        metric.update(torch.argmax(outputs, dim=1), labels, loss.item())

    # Assuming metric.value() returns a string for logging
    logger.info(f"Eval State: {metric.value()}")
    