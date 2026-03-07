

from loguru import logger
import torch

from utils.metric import Metric


def train(
    model: str,
    train_loader: list,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    device: str,
    epochs: int,
    learning_rate: float = 0.001,
):

    criterion = torch.nn.CrossEntropyLoss()
    metrics = ["acc"]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        # train the model on the training data for one epoch
        model.train()
        epoch_loss = 0.0
        
        for data, labels in train_loader:
            optimizer.zero_grad()
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            # NOTE: If the label is not one-hot encoded, the type of labels 
            # should be long for CrossEntropyLoss
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Log progress periodically
        # if (epoch + 1) % 5 == 0:
        evaluate(model, test_data, test_labels, device, metrics, criterion)
        avg_loss = epoch_loss / len(train_loader)

        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

@torch.no_grad()
def evaluate(model, data, labels, device, metrics, criterion):
    model.eval()
    metric = Metric(metrics)
    
    for session_id in range(data.shape[0]):
        session_data = data[session_id].to(device)
        session_labels = labels[session_id].to(device)
        
        outputs = model(session_data)
        
        # You might not need loss in eval, but if you do:
        loss = criterion(outputs, session_labels.long())
        
        # Update metric (Ensure your Metric class handles raw logits or add softmax here)
        metric.update(torch.argmax(outputs, dim=1), session_labels, loss.item())
    

    # Assuming metric.value() returns a string for logging
    logger.info(f"Eval State: {metric.value()}")
    
