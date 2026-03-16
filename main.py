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
from data.utils import merge_for_all_subjects, merge_for_one_subject, normalization_wrt_session, split_data_wrt_subjects, split_data_wrt_trials
# from train.training_TAHAG import train
from train.training import train

from utils.metric import Metric

# use typer to parse command line arguments and parse Traceback stack
app = typer.Typer(
    pretty_exceptions_show_locals=False,  # This hides the long list of variables
    # pretty_exceptions_short=True         # This makes the traceback even more concise
)

@app.command()
def main(
    model_name: Annotated[
        cli_enum.ModelName, typer.Option("-m", help="model name")
    ] = cli_enum.ModelName.SABER,
    dataset: Annotated[
        cli_enum.DatasetName, typer.Option(help="dataset name")
    ] = cli_enum.DatasetName.SEED,
    dataset_path: Annotated[
        str, typer.Option(help="path to the dataset")
    ] = "../data/SEED",
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
    batch_size: Annotated[int, typer.Option(help="batch size for training")] = 64,
    epochs: Annotated[int, typer.Option(help="number of epochs for training")] = 100,
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
        + f"\ndevice: {device}\nlogging level: {level}\ntask type: {task_type}"
        + f"\nsplit type: {split_type}\nbatch_size: {batch_size}\nepochs: {epochs}"
        + f"\ndata random: {data_random}"
    )

    data, labels, num_subjects, num_electrodes, num_features, num_classes = load_data(
        dataset, dataset_path
    )

    # data = normalization_wrt_session(data, type='min_max')

    num_sessions = len(labels)
    subject_ids = list(range(num_subjects))
    if data_random:
        shuffle(subject_ids)

    metric = Metric(num_subjects, num_sessions)

    logger.debug("num_sessions {} num_subjects {}", num_sessions, num_subjects)

    for session_id in range(num_sessions):
        for subject_id in subject_ids:
            if task_type == cli_enum.TaskTypeName.SUBJECT_DEPENDENT:
                train_data, train_labels, test_data, test_labels = (
                    split_data_wrt_trials(
                        data[session_id][subject_id], 
                        labels[session_id][subject_id], split_ratio, data_random)
                )
                # merge train data and labels
                train_data, train_labels = merge_for_one_subject(train_data, train_labels)
                # We keep session dimension for test data, 
                # since we want to test on all sessions separately.
                test_data, test_labels = merge_for_one_subject(test_data, test_labels)

                train_loader = DataLoader(
                    dataset=TensorDataset(
                        torch.tensor(train_data).float(),
                        torch.tensor(train_labels).float(),
                    ),
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=4,
                )

                model = MODEL[model_name](num_electrodes, num_features, num_classes).to(device)

                train(model, metric, train_loader, test_data, test_labels, batch_size, device, epochs, task_type, subject_id, session_id)

                logger.info("\n--------------> Finished training for subject {} session {} acc {:<.4f}",
                            subject_id, session_id, metric.accuracy[subject_id, session_id]
                            )

            else:
                # For subject-independent setting, we leave current subject out as test data
                # and merge the rest subjects' data as train data.
                train_data, train_labels, test_data, test_labels = (
                    split_data_wrt_subjects(
                        data[session_id], labels[session_id], subject_id)
                )

                train_data, train_labels = merge_for_all_subjects(train_data, train_labels)

                test_data, test_labels = merge_for_one_subject(test_data, test_labels)

                train_loader = DataLoader(
                    dataset=TensorDataset(
                        torch.tensor(train_data).float(),
                        torch.tensor(train_labels).float(),
                    ),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4
                )

                model = MODEL[model_name](num_electrodes, num_features, num_classes, domain_adaptation=True).to(device)


                train(model, metric, train_loader, test_data, test_labels, batch_size, device, epochs, task_type, subject_id, session_id)

                logger.info("\n--------------> Finished training for all subjects session {}, acc {:<.4f} on subject {}",
                            session_id, metric.accuracy[subject_id, session_id], subject_id
                            )
            
            if only_one_experiment:
                break


    logger.info("\n-----------> Finished training for all subjects!!!!")
    all_mean, all_std = metric.all_sessions_mean_acc()
    two_mean, two_std = metric.two_best_sessions_mean_acc()
    one_mean, one_std = metric.one_best_session_mean_acc()

    logger.info("\n all: mean {:<.4f} std {:<.4f}\ntwo: mean {:<.4f} std {:<.4f}\none: mean {:<.4f} std {:<.4f}\n",
                all_mean, all_std, two_mean, two_std, one_mean, one_std)


if __name__ == "__main__":
    app()