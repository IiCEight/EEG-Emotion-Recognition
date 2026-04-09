from random import shuffle
from typing import Annotated

import torch
import typer
import constant.CLI_arguments_enum as cli_enum
from loguru import logger

from config.logging import setUpLogger
from constant.model_map import MODEL
from data.dataloder import load_data
from data.utils import merge_and_split
from train.training_adann import train
from utils.metric import Metric
from utils.random_seed import setup_seed

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    dataset: Annotated[cli_enum.DatasetName, typer.Option(help='dataset name')] = cli_enum.DatasetName.SEED,
    dataset_path: Annotated[str, typer.Option(help='path to the dataset')] = '../data/SEED',
    cache_dir: Annotated[str | None, typer.Option(help='cache directory for loaded dataset (disabled if not set)')] = './cache',
    device: Annotated[str, typer.Option(help='device to run the model on')] = 'cuda',
    task_type: Annotated[cli_enum.TaskTypeName, typer.Option(help='type of experimental task (subject-dependent, subject-independent)')] = cli_enum.TaskTypeName.SUBJECT_INDEPENDENT,
    split_type: Annotated[cli_enum.SplitTypeName, typer.Option(help='type of data split (kfold, leave-one-subject-out)')] = cli_enum.SplitTypeName.LEAVE_ONE_SUBJECT_OUT,
    split_ratio: Annotated[float, typer.Option(help='ratio for train data size')] = 0.6,
    batch_size: Annotated[int, typer.Option(help='batch size for training')] = 128,
    epochs: Annotated[int, typer.Option(help='number of epochs for training')] = 60,
    data_random: Annotated[bool, typer.Option(help='whether to shuffle the data')] = False,
    only_one_experiment: Annotated[bool, typer.Option(help='whether to run only one experiment for debugging')] = False,
    only_one_session: Annotated[bool, typer.Option(help='whether to run only one session for debugging')] = True,
    random_seed: Annotated[int | None, typer.Option(help='random seed for reproducibility, None for no seed')] = 42,
    learning_rate: Annotated[float, typer.Option(help='learning rate for training')] = 0.001,
    early_stop_patience: Annotated[int, typer.Option(help='early stop after N epochs without test acc improvement (0=disabled)')] = 0,
    level: Annotated[cli_enum.LevelName, typer.Option('-l', help='level of severity for logging')] = cli_enum.LevelName.INFO,
):
    setUpLogger(level=level)
    setup_seed(random_seed)

    logger.info('CUDA Available: {}', torch.cuda.is_available())
    logger.info('Device Count: {}', torch.cuda.device_count())
    if torch.cuda.is_available():
        logger.info('GPU Name: {}', torch.cuda.get_device_name(0))

    data, labels, num_subjects, num_electrodes, num_features, num_classes = load_data(
        dataset_name=dataset,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
    )

    num_sessions = len(labels)
    subject_ids = list(range(num_subjects))
    if data_random:
        shuffle(subject_ids)

    metric = Metric(num_subjects, num_sessions)

    for session_id in range(num_sessions):
        for subject_id in subject_ids:
            setup_seed(random_seed)
            train_data, train_labels, test_data, test_labels = merge_and_split(
                data, labels, task_type, session_id, subject_id, split_ratio, data_random
            )

            model = MODEL[cli_enum.ModelName.ADANN](
                num_electrodes,
                num_features,
                num_classes,
            ).to(device)

            train(
                model,
                metric,
                train_data,
                train_labels,
                test_data,
                test_labels,
                batch_size,
                num_classes,
                device,
                epochs,
                task_type,
                subject_id,
                session_id,
                learning_rate,
                early_stop_patience,
            )

            logger.info('\n--------------> Finished training w.r.t. subject {} session {} acc {:<.4f}',
                        subject_id, session_id, metric.accuracy[subject_id, session_id])

            if only_one_experiment:
                break
        if only_one_session or only_one_experiment:
            break

    logger.info('\n-----------> Finished training for all subjects!!!!')
    all_mean, all_std = metric.all_sessions_mean_acc()
    two_mean, two_std = metric.two_best_sessions_mean_acc()
    one_mean, one_std = metric.one_best_session_mean_acc()

    logger.info('\nall: mean {:<.4f} std {:<.4f}\ntwo: mean {:<.4f} std {:<.4f}\none: mean {:<.4f} std {:<.4f}\n',
                all_mean, all_std, two_mean, two_std, one_mean, one_std)


if __name__ == '__main__':
    app()
