from typing import Annotated

import pickle
import random

import numpy as np
import torch
import typer
from einops import rearrange
from loguru import logger

import constant.CLI_arguments_enum as cli_enum
from config.logging import setUpLogger
from data.dataloder import load_data
from data.utils import merge_and_split, normalization_wrt_subject
from model.PCL_TDGCN import PCL
from model.PCL_SABER import PCL_SABER
from train import training_pcl
from utils.metric import Metric


def _set_seed_pcl(seed: int) -> None:
    """Match main_PCL.set_seed exactly (do NOT touch cudnn.enabled)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    dataset: Annotated[str, typer.Option("-d", help="dataset name")] = "SEED",
    dataset_path: Annotated[str, typer.Option(help="path to dataset")] = "../data/SEED",
    cache_dir: Annotated[str | None, typer.Option(help="cache directory")] = "./cache",
    device: Annotated[str, typer.Option(help="device")] = "cuda:0",
    batch_size: Annotated[int, typer.Option(help="batch size")] = 32,
    epochs: Annotated[int, typer.Option(help="epochs")] = 1000,
    lr: Annotated[float, typer.Option(help="learning rate")] = 0.001,
    weight_decay: Annotated[float, typer.Option(help="weight decay")] = 0.001,
    seed: Annotated[int, typer.Option(help="random seed")] = 200,
    eval_interval: Annotated[int, typer.Option(help="evaluate every N epochs")] = 1,
    early_stop_patience: Annotated[int, typer.Option(help="early stop patience (0=disabled)")] = 1000,
    layers: Annotated[int, typer.Option(help="MHGCN layers")] = 2,
    hidden_1: Annotated[int, typer.Option(help="encoder hidden dim 1")] = 256,
    hidden_2: Annotated[int, typer.Option(help="encoder hidden dim 2 / feature dim")] = 64,
    only_one_experiment: Annotated[bool, typer.Option(help="run one subject only (debug)")] = False,
    only_one_session: Annotated[bool, typer.Option(help="run one session only")] = True,
    use_saber_encoder: Annotated[bool, typer.Option(help="replace MHGCN encoder with Saber's FeatureExtractor")] = False,
    direct_cache: Annotated[str | None, typer.Option(help="load dataset directly from this .pkl file (bypasses load_data)")] = None,
    level: Annotated[cli_enum.LevelName, typer.Option("-l", help="log level")] = cli_enum.LevelName.INFO,
):
    """PCL-TDGCN subject-independent training entry point."""
    setUpLogger(level=level)
    _set_seed_pcl(seed)

    logger.info(
        f'Launching....\nmodel_name: PCL\ndataset: {dataset}\ndataset_path: {dataset_path}'
        + f'\ncache_dir: {cache_dir}'
        + f'\ndevice: {device}\nlogging level: {level}'
        + f'\nbatch_size: {batch_size}\nepochs: {epochs}'
        + f'\nrandom seed: {seed}'
        + f'\nonly one experiment: {only_one_experiment}\nonly one session: {only_one_session}'
        + f'\nlearning rate: {lr}\nearly stop patience: {early_stop_patience}'
    )

    if direct_cache is not None:
        with open(direct_cache, "rb") as f:
            data, labels, num_subjects, num_electrodes, num_features, num_classes = pickle.load(f)["result"]
    else:
        data, labels, num_subjects, num_electrodes, num_features, num_classes = load_data(
            dataset_name=dataset,
            dataset_path=dataset_path,
            cache_dir=cache_dir,
        )

    normalization_wrt_subject(data, band_major=True)

    num_sessions = len(labels)
    metric = Metric(num_subjects, num_sessions)

    for session_id in range(num_sessions):
        for subject_id in range(num_subjects):
            _set_seed_pcl(seed)

            train_data, train_labels, test_data, test_labels = merge_and_split(
                data, labels,
                task_type=cli_enum.TaskTypeName.SUBJECT_INDEPENDENT,
                session_id=session_id,
                subject_id=subject_id,
                split_ratio=0.6,
                data_random=False,
            )

            # (N, 62, 5) → (N, 5, 62) → (N, 310) band-major (already normalized per subject)
            train_data = rearrange(train_data, "s c f -> s f c").reshape(-1, 310).astype("float32")
            test_data = rearrange(test_data, "s c f -> s f c").reshape(-1, 310).astype("float32")

            source_num = train_data.shape[0]
            target_num = test_data.shape[0]

            model_cls = PCL_SABER if use_saber_encoder else PCL
            model = model_cls(
                in_planes=[num_features, num_electrodes],
                layers=layers,
                hidden_1=hidden_1,
                hidden_2=hidden_2,
                num_of_class=num_classes,
                device=device,
                source_num=source_num,
                target_num=target_num,
            ).to(device)

            training_pcl.train(
                model=model,
                metric=metric,
                train_data=train_data,
                train_labels=train_labels,
                test_data=test_data,
                test_labels=test_labels,
                batch_size=batch_size,
                num_classes=num_classes,
                device=device,
                epochs=epochs,
                subject_id=subject_id,
                session_id=session_id,
                learning_rate=lr,
                weight_decay=weight_decay,
                eval_interval=eval_interval,
                early_stop_patience=early_stop_patience,
            )

            logger.info(
                "Done subj={} sess={} acc={:.4f}",
                subject_id, session_id,
                metric.accuracy[subject_id, session_id],
            )

            if only_one_experiment:
                break
        if only_one_session or only_one_experiment:
            break

    mean, std = metric.one_best_session_mean_acc()
    logger.info("One best session acc: mean={:.4f} std={:.4f}", mean, std)


if __name__ == "__main__":
    app()
