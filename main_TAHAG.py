from random import shuffle
from typing import Annotated
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import torch
from config.logging import setUpLogger

import typer
import constant.CLI_arguments_enum as cli_enum
from loguru import logger

from constant.model_map import MODEL
from data.dataloder import load_data
from data.utils import merge_for_all_subjects, merge_for_one_subject, split_data
from train.training_TAHAG import train
# from train.training import train

from utils.graphConstructionFromStandard import get_adj_from_standard
from utils.metric import Metric

# use typer to parse command line arguments and parse Traceback stack
app = typer.Typer(
    pretty_exceptions_show_locals=False,  # This hides the long list of variables
    # pretty_exceptions_short=True         # This makes the traceback even more concise
)

@app.command()
def main(
    model_name: Annotated[
        cli_enum.ModelName, typer.Argument(help="model name")
    ] = cli_enum.ModelName.TAHAG,
    dataset: Annotated[
        cli_enum.DatasetName, typer.Argument(help="dataset name")
    ] = cli_enum.DatasetName.SEED,
    dataset_path: Annotated[
        str, typer.Option(help="path to the dataset")
    ] = "../data/SEED",
    cache_dir: Annotated[str | None, typer.Option(
        help="cache directory for loaded dataset (disabled if not set)")] = None,
    device: Annotated[str, typer.Option(help="device to run the model on")] = "cuda",
    sample_length: Annotated[
        int, typer.Option(help="length of data points in each sample")
    ] = 1,
    stride: Annotated[int, typer.Option(help="stride for segmenting data")] = 128,
    label_type: Annotated[
        str, typer.Option(help="type of label to use (valence, arousal)")
    ] = "valence",
    task_type: Annotated[
        cli_enum.TaskTypeName,
        typer.Option(
            help="type of experimental task (subject-dependent, subject-independent)"
        ),
    ] = cli_enum.TaskTypeName.SUBJECT_INDEPENDENT,
    split_type: Annotated[
        cli_enum.SplitTypeName,
        typer.Option(
            help="type of data split (kfold, leave-one-subject-out)"
        ),
    ] = cli_enum.SplitTypeName.LEAVE_ONE_SUBJECT_OUT,
    split_ratio: Annotated[
        float, typer.Option(help="ratio for train data size")
    ] = 0.6,
    batch_size: Annotated[int, typer.Option(help="batch size for training")] = 128,
    epochs: Annotated[int, typer.Option(help="number of epochs for training")] = 200,
    data_random: Annotated[bool, typer.Option(help="whether to shuffle the data")] = False,
    only_one_experiment: Annotated[bool, typer.Option(help="whether to run only one experiment for debugging")] = True,
    level: Annotated[
        cli_enum.LevelName, typer.Option("-l", help="level of severity for logging")
    ] = cli_enum.LevelName.INFO
):
    """
    Welcome!

    Use "--help" option to see usage information.
    """

    # ------------------ set up logger ------------------
    setUpLogger(level=level)

    logger.info("CUDA Available: {}", torch.cuda.is_available())
    logger.info("Device Count: {}", torch.cuda.device_count())

    if torch.cuda.is_available():
        logger.info("GPU Name: {}", torch.cuda.get_device_name(0))

    logger.info(
        f"Launching....\nmodel_name: {model_name}\ndataset: {dataset}\ndataset_path: {dataset_path}"
        + f"\ncache_dir: {cache_dir}"
        + f"\ndevice: {device}\nlogging level: {level}\ntask type: {task_type}"
        + f"\nsplit type: {split_type}\nbatch_size: {batch_size}\nepochs: {epochs}"
        + f"\ndata random: {data_random}"
    )

    data, labels, num_subjects, num_electrodes, num_features, num_classes = load_data(
        dataset_name=dataset,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        sample_length=sample_length,
        stride=stride,
        label_type=label_type,
    )

    subject_ids = list(range(num_subjects))
    if data_random:
        shuffle(subject_ids)

    metric = Metric(num_subjects)

    for subject_id in subject_ids:
        if task_type == cli_enum.TaskTypeName.SUBJECT_DEPENDENT:
            train_data, train_labels, test_data, test_labels = (
                split_data(data[subject_id], labels[subject_id], split_ratio, data_random)
            )
            # merge train data and labels
            train_data, train_labels = merge_for_one_subject(train_data, train_labels)
            # We keep session dimension for test data, 
            # since we want to test on all sessions separately.
            test_data, test_labels = merge_for_one_subject(test_data, test_labels, keep_session_dim=True)

            train_loader = DataLoader(
                dataset=TensorDataset(
                    torch.tensor(train_data).float(),
                    torch.tensor(train_labels).float(),
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            adj_matrix = get_adj_from_standard()

            model = MODEL[model_name](num_electrodes, num_features, num_classes).to(device)

            train(model, metric, train_loader, test_data, test_labels, batch_size, device, epochs, task_type, subject_id)

            logger.info(f"\n--------------> Finished training for subject {subject_id}")
            logger.info("Best all sessions : {:<.4f}, "
                        "two sessions: {:<.4f}, one session : {:<.4f}",
                        metric.best_subject_avg_acc[subject_id], 
                        metric.best_two_sessions_subject_avg_acc[subject_id], 
                        metric.best_one_session_subject_avg_acc[subject_id]
                        )

        else:
            # For subject-independent setting, we leave current subject out as test data
            # and merge the rest subjects' data as train data.
            mask = np.ones(num_subjects, dtype=bool)
            mask[subject_id] = False
            train_data = data[mask]
            train_labels = labels[mask]
            test_data = data[subject_id]
            test_labels = labels[subject_id]

            train_data, train_labels = merge_for_all_subjects(train_data, train_labels)
            # We keep session dimension for test data, 
            # since we want to test on all sessions separately.
            test_data, test_labels = merge_for_one_subject(test_data, test_labels, keep_session_dim=True)

            train_loader = DataLoader(
                dataset=TensorDataset(
                    torch.tensor(train_data).float(),
                    torch.tensor(train_labels).float(),
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
            adj_matrix = get_adj_from_standard()

            model = MODEL[model_name](num_electrodes, num_features, num_classes, domain_adaptation=True).to(device)


            train(model, metric, train_loader, test_data, test_labels, batch_size, device, epochs, task_type, subject_id)

            logger.info(f"\n--------------> Finished training for subject {subject_id}")
            logger.info("Best all sessions : {:<.4f}, "
                        "two sessions: {:<.4f}, one session : {:<.4f}",
                        metric.best_subject_avg_acc[subject_id], 
                        metric.best_two_sessions_subject_avg_acc[subject_id], 
                        metric.best_one_session_subject_avg_acc[subject_id]
                        )
        
        if only_one_experiment:
            break


    logger.info(f"\n-----------> Finished training for all subjects!!!!")
    all_sessions, two, one = metric.best_all_subjects_avg_acc() 
    logger.info("Best all subjects avg acc: {:<.4f}, "
                "two sessions: {:<.4f}, one session : {:<.4f}",
                all_sessions, two, one
                )


if __name__ == "__main__":
    app()