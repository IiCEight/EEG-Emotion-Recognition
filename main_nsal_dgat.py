from random import shuffle
from typing import Annotated

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

import typer
import constant.CLI_arguments_enum as cli_enum
from config.logging import setUpLogger
from constant.model_map import MODEL
from data.dataloder import load_data
from data.utils import (
    merge_for_all_subjects,
    merge_for_one_subject,
    split_data_wrt_subjects,
    split_data_wrt_trials,
)
from train.training_nsal_dgat import train
from utils.metric import Metric


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    dataset: Annotated[
        cli_enum.DatasetName, typer.Option(help="dataset name")
    ] = cli_enum.DatasetName.SEED,
    dataset_path: Annotated[
        str, typer.Option(help="path to the dataset")
    ] = "../data/SEED",
    device: Annotated[str, typer.Option(help="device to run the model on")] = "cuda:0",
    task_type: Annotated[
        cli_enum.TaskTypeName,
        typer.Option(
            help="type of experimental task (subject-dependent, subject-independent)"
        ),
    ] = cli_enum.TaskTypeName.SUBJECT_INDEPENDENT,
    split_ratio: Annotated[
        float, typer.Option(help="ratio for train data size")
    ] = 0.6,
    batch_size: Annotated[int, typer.Option(help="batch size for training")] = 128,
    epochs: Annotated[int, typer.Option(help="number of epochs for training")] = 60,
    data_random: Annotated[bool, typer.Option(help="whether to shuffle the data")] = False,
    only_one_experiment: Annotated[
        bool, typer.Option(help="whether to run only one experiment for debugging")
    ] = False,
    level: Annotated[
        cli_enum.LevelName, typer.Option("-l", help="level of severity for logging")
    ] = cli_enum.LevelName.INFO,
):
    setUpLogger(level=level)

    model_name = cli_enum.ModelName.NSAL_DGAT

    logger.info("CUDA Available: {}", torch.cuda.is_available())
    logger.info("Device Count: {}", torch.cuda.device_count())
    if torch.cuda.is_available():
        logger.info("GPU Name: {}", torch.cuda.get_device_name(0))

    logger.info(
        f"Launching....\nmodel_name: {model_name}\ndataset: {dataset}\ndataset_path: {dataset_path}"
        + f"\ndevice: {device}\nlogging level: {level}\ntask type: {task_type}"
        + f"\nbatch_size: {batch_size}\nepochs: {epochs}\ndata random: {data_random}"
    )

    data, labels, num_subjects, num_electrodes, num_features, num_classes = load_data(
        dataset, dataset_path
    )

    num_sessions = len(labels)
    subject_ids = list(range(num_subjects))
    if data_random:
        shuffle(subject_ids)

    logger.debug("num_sessions {} num_subjects {}", num_sessions, num_subjects)

    if task_type == cli_enum.TaskTypeName.SUBJECT_INDEPENDENT:
        # Match original LibEER protocol: merge sessions first, then LOSO on subjects.
        merged_data = [[] for _ in range(num_subjects)]
        merged_labels = [[] for _ in range(num_subjects)]
        for session_id in range(num_sessions):
            for subject_id in range(num_subjects):
                merged_data[subject_id].extend(data[session_id][subject_id])
                merged_labels[subject_id].extend(labels[session_id][subject_id])

        metric = Metric(num_subjects, 1)

        for subject_id in subject_ids:
            train_data, train_labels, test_data, test_labels = split_data_wrt_subjects(
                merged_data, merged_labels, subject_id
            )
            train_data, train_labels = merge_for_all_subjects(train_data, train_labels)
            test_data, test_labels = merge_for_one_subject(test_data, test_labels)

            train_tensor = torch.tensor(train_data).float()
            label_tensor = torch.tensor(train_labels).float()
            train_dataset = TensorDataset(
                train_tensor,
                torch.arange(train_tensor.size(0), dtype=torch.long),
                label_tensor,
            )

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )

            model = MODEL[model_name](
                num_electrodes,
                num_features,
                num_classes,
                domain_adaptation=True,
                source_num=len(train_data),
                device=device,
            ).to(device)

            train(
                model,
                metric,
                train_loader,
                test_data,
                test_labels,
                batch_size,
                device,
                epochs,
                task_type,
                subject_id,
                0,
            )

            logger.info(
                "\n--------------> Finished merged-session LOSO acc {:<.4f} on subject {}",
                metric.accuracy[subject_id, 0],
                subject_id,
            )

            if only_one_experiment:
                break
    else:
        metric = Metric(num_subjects, num_sessions)
        for session_id in range(num_sessions):
            for subject_id in subject_ids:
                train_data, train_labels, test_data, test_labels = split_data_wrt_trials(
                    data[session_id][subject_id],
                    labels[session_id][subject_id],
                    split_ratio,
                    data_random,
                )

                train_data, train_labels = merge_for_one_subject(train_data, train_labels)
                test_data, test_labels = merge_for_one_subject(test_data, test_labels)

                train_tensor = torch.tensor(train_data).float()
                label_tensor = torch.tensor(train_labels).float()
                train_dataset = TensorDataset(
                    train_tensor,
                    torch.arange(train_tensor.size(0), dtype=torch.long),
                    label_tensor,
                )

                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=4,
                )

                model = MODEL[model_name](
                    num_electrodes,
                    num_features,
                    num_classes,
                    domain_adaptation=False,
                    source_num=len(train_data),
                    device=device,
                ).to(device)

                train(
                    model,
                    metric,
                    train_loader,
                    test_data,
                    test_labels,
                    batch_size,
                    device,
                    epochs,
                    task_type,
                    subject_id,
                    session_id,
                )

                logger.info(
                    "\n--------------> Finished training for subject {} session {} acc {:<.4f}",
                    subject_id,
                    session_id,
                    metric.accuracy[subject_id, session_id],
                )

                if only_one_experiment:
                    break
            if only_one_experiment:
                break

    logger.info("\n-----------> Finished training for all subjects!!!!")
    all_mean, all_std = metric.all_sessions_mean_acc()
    two_mean, two_std = metric.two_best_sessions_mean_acc()
    one_mean, one_std = metric.one_best_session_mean_acc()

    logger.info(
        "\n all: mean {:<.4f} std {:<.4f}\ntwo: mean {:<.4f} std {:<.4f}\none: mean {:<.4f} std {:<.4f}\n",
        all_mean,
        all_std,
        two_mean,
        two_std,
        one_mean,
        one_std,
    )


if __name__ == "__main__":
    app()
