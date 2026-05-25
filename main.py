from random import shuffle
from typing import Annotated
import torch
from config.logging import setUpLogger

import typer
import constant.CLI_arguments_enum as cli_enum
from loguru import logger

from constant.model_map import MODEL
from data.dataloder import load_data
from data.utils import merge_and_split, normalization_wrt_subject
from model.saber import Saber
from model.saber_t import SaberT
from train.training import train
import train.training_saber_t as training_saber_t

from utils.metric import Metric
from utils.random_seed import setup_seed
from utils.failure_sample import build_test_metadata, init_failure_log

app = typer.Typer(
    pretty_exceptions_show_locals=False,
)


@app.command()
def main(
    model_name: Annotated[
        cli_enum.ModelName, typer.Option('-m', help='model name')
    ] = cli_enum.ModelName.OPTA,
    dataset: Annotated[
        cli_enum.DatasetName, typer.Option(help='dataset name')
    ] = cli_enum.DatasetName.SEED,
    dataset_path: Annotated[
        str, typer.Option(help='path to the dataset')
    ] = '../data/SEED',
    cache_dir: Annotated[str | None, typer.Option(
        help='cache directory for loaded dataset (disabled if not set)')] = './cache',
    device: Annotated[str, typer.Option(
        help='device to run the model on')] = 'cuda',
    sample_length: Annotated[
        int, typer.Option(help='length of data points in each sample')
    ] = 1,
    stride: Annotated[int, typer.Option(
        help='stride for segmenting data')] = 1,
    label_type: Annotated[
        str, typer.Option(help='type of label to use (valence, arousal)')
    ] = 'valence',
    task_type: Annotated[
        cli_enum.TaskTypeName,
        typer.Option(help='type of experimental task (subject-dependent, subject-independent)'),
    ] = cli_enum.TaskTypeName.SUBJECT_INDEPENDENT,
    split_type: Annotated[
        cli_enum.SplitTypeName,
        typer.Option(help='type of data split (kfold, leave-one-subject-out)'),
    ] = cli_enum.SplitTypeName.LEAVE_ONE_SUBJECT_OUT,
    split_ratio: Annotated[float, typer.Option(help='ratio for train data size')] = 0.6,
    batch_size: Annotated[int, typer.Option(help='batch size for training')] = 32,
    epochs: Annotated[int, typer.Option(help='number of epochs for training')] = 1000,
    data_random: Annotated[bool, typer.Option(help='whether to shuffle the data')] = False,
    only_one_experiment: Annotated[bool, typer.Option(help='whether to run only one experiment for debugging')] = False,
    only_one_session: Annotated[bool, typer.Option(help='whether to run only one session for debugging')] = True,
    random_seed: Annotated[int | None, typer.Option(help='random seed for reproducibility, None for no seed (i.e., random)')] = 42,
    learning_rate: Annotated[float, typer.Option(help='learning rate for training')] = 0.001,
    early_stop_patience: Annotated[int, typer.Option(help='early stop after N epochs without test acc improvement (0 = disabled)')] = 0,
    failure_log: Annotated[str | None, typer.Option("-f", help='path to CSV file for logging misclassified samples (disabled if not set)')] = None,
    use_gcn: Annotated[bool, typer.Option(help='use GCN feature extractor for SABER (False = flat MLP, faster)')] = True,
    time_steps: Annotated[int, typer.Option(help='number of consecutive DE windows per sample for SABER_T (requires --sample-length N)')] = 2,
    trim_trial_start_pct: Annotated[float, typer.Option(help='discard the first X%% of each trial at load time (0 = disabled)')] = 0.0,
    opta_pool_capacity: Annotated[int, typer.Option(
        help='OPTA: target prototype pool capacity'
    )] = 256,
    xconf_ramp_epochs: Annotated[int, typer.Option(
        help='OPTA: epochs over which xconf lam3 ramps from 0 to 0.2'
    )] = 200,
    sinkhorn_warmup_epochs: Annotated[int, typer.Option(
        help='OPTA: epochs to keep Sinkhorn assignments detached before allowing gradients'
    )] = 100,
    level: Annotated[cli_enum.LevelName, typer.Option('-l', help='level of severity for logging')] = cli_enum.LevelName.INFO,
):
    """Welcome! Use --help option to see usage information."""

    setUpLogger(level=level)
    setup_seed(random_seed)

    logger.info('CUDA Available: {}', torch.cuda.is_available())
    logger.info('Device Count: {}', torch.cuda.device_count())
    if torch.cuda.is_available():
        logger.info('GPU Name: {}', torch.cuda.get_device_name(0))

    logger.info(
        f'Launching....\nmodel_name: {model_name}\ndataset: {dataset}\ndataset_path: {dataset_path}'
        + f'\ncache_dir: {cache_dir}'
        + f'\ndevice: {device}\nlogging level: {level}\ntask type: {task_type}'
        + f'\nsplit type: {split_type}\nbatch_size: {batch_size}\nepochs: {epochs}'
        + f'\ndata random: {data_random}\nrandom seed: {random_seed}\nfailure_log: {failure_log}'
        + f'\nonly one experiment: {only_one_experiment}\nonly one session: {only_one_session}'
        + f'\nlearning rate: {learning_rate}\nearly stop patience: {early_stop_patience}'
        + f'\nuse_gcn: {use_gcn}\ntime_steps: {time_steps}\ntrim_trial_start_pct: {trim_trial_start_pct}'
        + f'\nopta_pool_capacity: {opta_pool_capacity}\nsinkhorn_warmup_epochs: {sinkhorn_warmup_epochs}'
    )

    data, labels, num_subjects, num_electrodes, num_features, num_classes = load_data(
        dataset_name=dataset,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        sample_length=sample_length,
        stride=stride,
        trim_trial_start_pct=trim_trial_start_pct,
    )

    # This help a lot.
    normalization_wrt_subject(data)

    num_sessions = len(labels)
    subject_ids = list(range(num_subjects))
    if data_random:
        shuffle(subject_ids)

    metric = Metric(num_subjects, num_sessions)

    if failure_log is not None:
        init_failure_log(failure_log)

    logger.debug('num_sessions {} num_subjects {}', num_sessions, num_subjects)

    skip = 0

    for session_id in range(num_sessions):
        for subject_id in subject_ids:
            if subject_id < skip:
                continue
            setup_seed(random_seed)

            # Build per-sample metadata before merging discards trial indices.
            # For LOSO subject-independent, the test subject's raw trials are at
            # data[session_id][subject_id].
            test_metadata = None
            if failure_log is not None and task_type == cli_enum.TaskTypeName.SUBJECT_INDEPENDENT:
                test_metadata = build_test_metadata(data[session_id][subject_id])

            train_data, train_labels, test_data, test_labels = merge_and_split(
                data, labels, task_type, session_id, subject_id, split_ratio, data_random,
                time_steps=time_steps if model_name == cli_enum.ModelName.SABER_T else 1,
            )

            if model_name == cli_enum.ModelName.SABER_T:
                model = SaberT(
                    num_electrodes, num_features, time_steps, num_classes, use_gcn=use_gcn,
                ).to(device)
                training_saber_t.train(
                    model, metric, train_data, train_labels, test_data, test_labels,
                    batch_size, num_classes, device, epochs, task_type, subject_id,
                    session_id, learning_rate, early_stop_patience,
                    test_metadata=test_metadata, failure_log_path=failure_log,
                )
            elif model_name == cli_enum.ModelName.OPTA:
                from model.opta import OPTA
                import train.training_opta as training_opta

                model = OPTA(
                    num_electrodes=num_electrodes,
                    in_features=num_features,
                    num_classes=num_classes,
                    use_gcn=use_gcn,
                    max_iter=epochs,
                    pool_capacity=opta_pool_capacity,
                    sinkhorn_warmup_epochs=sinkhorn_warmup_epochs,
                ).to(device)
                training_opta.train(
                    model, metric, train_data, train_labels, test_data, test_labels,
                    batch_size, num_classes, device, epochs, task_type, subject_id,
                    session_id, learning_rate, early_stop_patience,
                    xconf_ramp_epochs=xconf_ramp_epochs,
                    test_metadata=test_metadata, failure_log_path=failure_log,
                )
            else:
                model = Saber(
                    num_electrodes, num_features, num_classes, use_gcn=use_gcn,
                ).to(device)

                train(model, metric, train_data, train_labels, test_data, test_labels,
                                  batch_size, num_classes, device, epochs, task_type, subject_id,
                                  session_id, learning_rate, early_stop_patience,
                                  test_metadata=test_metadata, failure_log_path=failure_log)


            logger.info("\n--------------> Finished training w.r.t. subject {} session {} acc {:<.4f}",
                        subject_id, session_id, metric.accuracy[subject_id, session_id]
                        )

            if only_one_experiment:
                break
        if only_one_session or only_one_experiment:
            break

    logger.info('\n-----------> Finished training for all subjects!!!!')
    one_mean, one_std = metric.one_best_session_mean_acc()

    logger.info(
        'One best session acc: mean {:<.4f} std {:<.4f}\n',
        one_mean,
        one_std,
    )


if __name__ == '__main__':
    app()
