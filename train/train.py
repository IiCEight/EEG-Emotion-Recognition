from loguru import logger
import torch
from torch.utils.data import DataLoader, TensorDataset
from constant import CLI_arguments_enum
from constant.model_map import MODEL
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
    criterion = torch.nn.CrossEntropyLoss()
    metrics = ["acc"]

    # ==================================================
    # CASE 1: Subject-Dependent (32 Separate Models)
    # ==================================================
    if task_type == CLI_arguments_enum.TaskTypeName.SUBJECT_DEPENDENT:
        # train a separate model for each subject
        for s_i in range(num_subjects):
            logger.info(f"--- Starting Training for Subject {s_i} ---")

            # 1. Initialize FRESH model for this subject
            model = MODEL[model_name](num_electrodes, num_features, num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # NOTE: we need initialize a new model for each subject

            # 2. Create DataLoaders (FIXED: Added .float() and .float())
            train_loader = DataLoader(
                dataset=TensorDataset(
                    torch.tensor(split_dataset[0][s_i]).float(), 
                    torch.tensor(split_dataset[1][s_i]).float()
                ),
                batch_size=batch_size, shuffle=True, num_workers=4
            )
            val_loader = DataLoader(
                dataset=TensorDataset(
                    torch.tensor(split_dataset[2][s_i]).float(), 
                    torch.tensor(split_dataset[3][s_i]).float()
                ),
                batch_size=batch_size, shuffle=False, num_workers=4
            )

            # 3. Training Loop
            for epoch in range(epochs):
                # train the model on the training data for one epoch
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} for subject {s_i}")
                model.train()
                epoch_loss = 0.0
                
                for batch in train_loader:
                    optimizer.zero_grad() # Correct placement
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
                logger.info(f"Sub {s_i} | Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

            # 4. Final Evaluation for this subject
            # (You might want to return these results to calculate an average later)
            # evaluate(model, val_loader, device, metrics, criterion)

    # ==================================================
    # CASE 2: Subject-Independent (1 Global Model)
    # ==================================================
    else:
        logger.info("--- Starting Training for Subject Independent Model ---")
        
        model = MODEL[model_name](num_electrodes, num_features, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # train a single model on all the data
        # construct pytorch dataloaders for all the data
        train_loader = DataLoader(
            dataset=TensorDataset(
                torch.tensor(split_dataset[0]).float(), 
                torch.tensor(split_dataset[1]).float()
            ),
            batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            dataset=TensorDataset(
                torch.tensor(split_dataset[2]).float(), 
                torch.tensor(split_dataset[3]).float()
            ),
            batch_size=batch_size, shuffle=False, num_workers=4
        )

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # if (epoch + 1) % 5 == 0:
            # Run validation inside the loop to monitor overfitting
            evaluate(model, val_loader, device, metrics, criterion)
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")


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
    