from loguru import logger
import torch
from constant import CLI_arguments_enum
from torch.utils.data import DataLoader,TensorDataset

from constant.model_map import MODEL
from utils.metric import Metric

def train(
    model_name: str,
    split_dataset:list,
    num_subjects:int,
    num_electrodes:int,
    num_features:int,
    num_classes:int,
    device:str,
    task_type:str,
    batch_size:int,
    epochs:int,
    learning_rate:float = 0.001,
):
    """
    split_dataset: the output of load_data function, which is a list
        containing the split data and labels for training, validation and testing.

    This function will train the model on the training data and evaluate on the validation
    """
    object_function = torch.nn.CrossEntropyLoss()
    metrics = ["acc"]

    if task_type == CLI_arguments_enum.TaskTypeName.SUBJECT_DEPENDENT:
        # train a separate model for each subject
        for s_i in range(num_subjects):
            logger.info(f"--- Starting Training for Subject {s_i} ---")

            model = MODEL[model_name](num_electrodes, num_features, num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # NOTE: we need initialize a new model for each subject


            # construct pytorch dataloaders for the subject s_i
            train_loader = DataLoader(
                dataset=TensorDataset(torch.tensor(split_dataset[0][s_i]), 
                                      torch.tensor(split_dataset[1][s_i]).long()),         # The dataset to load from
                batch_size=batch_size,           # How many samples per batch
                shuffle=True,            # Mix the data every epoch
                num_workers=4,           # Use 4 CPU cores to load data in parallel
            )
            val_loader = DataLoader(
                dataset=TensorDataset(torch.tensor(split_dataset[2][s_i]), 
                                      torch.tensor(split_dataset[3][s_i]).long()),         # The dataset to load from
                batch_size=batch_size,           # How many samples per batch
                shuffle=False,           # Don't mix validation data
                num_workers=4,           # Use 4 CPU cores to load data in parallel
            )

            for epoch in range(epochs):
                # train the model on the training data for one epoch
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{epochs} for subject {s_i}")
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    loss = object_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
            # evaluate the model on the validation data for each subject
            
            evaluate(model, val_loader, device, metrics, object_function)
    else:

        model = MODEL[model_name](num_electrodes, num_features, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # train a single model on all the data
        # construct pytorch dataloaders for all the data
        train_loader = DataLoader(
            dataset=TensorDataset(torch.tensor(split_dataset[0]), 
                                  torch.tensor(split_dataset[1]).long()),         # The dataset to load from
            batch_size=batch_size,           # How many samples per batch
            shuffle=True,            # Mix the data every epoch
            num_workers=4,           # Use 4 CPU cores to load data in parallel
        )
        val_loader = DataLoader(
            dataset=TensorDataset(torch.tensor(split_dataset[2]), 
                                  torch.tensor(split_dataset[3]).long()),         # The dataset to load from
            batch_size=batch_size,           # How many samples per batch
            shuffle=False,              # Don't mix validation data
            num_workers=4,              # Use 4 CPU cores to load data in parallel
        )

        for epoch in range(epochs):
            # train the model on the training data for one epoch
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}")
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = object_function(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # evaluate the model on the validation data
        evaluate(model, val_loader, device, metrics, object_function)


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion):
    model.eval()
    # create Metric object
    metric = Metric(metrics)
    for idx, (samples, targets) in data_loader:
        # load the samples into the device
        samples = samples.to(device)
        targets = targets.to(device)

        # perform emotion recognition
        outputs = model(samples)

        # calculate the loss value
        loss = criterion(outputs, targets)
        # one hot code
        # loss = criterion(outputs, targets)
        metric.update(torch.argmax(outputs, dim=1), targets, loss.item())

    logger.info(" eval state: " + metric.value())
    return metric.values