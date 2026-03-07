from typing import Annotated

import torch
from config.logging import setUpLogger

import typer
import constant.CLI_arguments_enum as cli_enum
from loguru import logger

from constant.model_map import MODEL
from data.dataloder import load_data
from data.merge import merge
from train.train import train

# use typer to parse command line arguments and parse Traceback stack
app = typer.Typer(
    pretty_exceptions_show_locals=False,  # This hides the long list of variables
    # pretty_exceptions_short=True         # This makes the traceback even more concise
)

@app.command()
def main(
    model: Annotated[
        cli_enum.ModelName, typer.Argument(help="model name")
    ] = cli_enum.ModelName.TAHAG,
    dataset: Annotated[
        cli_enum.DatasetName, typer.Argument(help="dataset name")
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
            help="type of data split (kfold, leave-one-subject-out, train-test-validation)"
        ),
    ] = cli_enum.SplitTypeName.TRAIN_TEST_VALIDATION,
    batch_size: Annotated[int, typer.Option(help="batch size for training")] = 32,
    epochs: Annotated[int, typer.Option(help="number of epochs for training")] = 20,
    level: Annotated[
        cli_enum.LevelName, typer.Option("-l", help="level of severity for logging")
    ] = cli_enum.LevelName.INFO,
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
        f"Launching....\nmodel: {model} dataset: {dataset} dataset_path: {dataset_path}"
        + f"device: {device} logging level: {level} task type: {task_type}"
        + f"split type: {split_type}"
    )

    data, labels, num_subjects, num_electrodes, num_features, num_classes = load_data(
        dataset, dataset_path
    )

    merge(data, labels)


    exit(0)


    train(
        model,
        split_dataset,
        num_subjects,
        num_electrodes,
        num_features,
        num_classes,
        device,
        task_type,
        batch_size,
        epochs,
    )


if __name__ == "__main__":
    app()