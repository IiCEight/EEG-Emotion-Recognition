from typing import Annotated
from config.logging import setUpLogger

import typer
import constant.CLI_arguments_enum as cli_enum
from loguru import logger

from constant.model_map import MODEL
from data.dataloder import load_data
from train.train import train


def main(
    model: Annotated[
        cli_enum.ModelName, typer.Argument(help="model name")
    ] = cli_enum.ModelName.EEGNET,
    dataset: Annotated[
        cli_enum.DatasetName, typer.Argument(help="dataset name")
    ] = cli_enum.DatasetName.DEAP,
    dataset_path: Annotated[
        str, typer.Option(help="path to the dataset")
    ] = "../LibEER/data/DEAP",
    device: Annotated[str, typer.Option(help="device to run the model on")] = "cpu",
    level: Annotated[
        cli_enum.LevelName, typer.Option(help="level of severity for logging")
    ] = cli_enum.LevelName.DEBUG,
    sample_length: Annotated[
        int, typer.Option(help="length of data points in each sample")
    ] = 128,
    stride: Annotated[int, typer.Option(help="stride for segmenting data")] = 128,
    label_type: Annotated[
        str, typer.Option(help="type of label to use (valence, arousal)")
    ] = "valence",
    task_type: Annotated[
        cli_enum.TaskTypeName,
        typer.Option(
            help="type of experimental task (subject-dependent, subject-independent)"
        ),
    ] = cli_enum.TaskTypeName.SUBJECT_DEPENDENT,
    split_type: Annotated[
        cli_enum.SplitTypeName,
        typer.Option(
            help="type of data split (kfold, leave-one-subject-out, train-test-validation)"
        ),
    ] = cli_enum.SplitTypeName.TRAIN_TEST_VALIDATION,
    batch_size: Annotated[int, typer.Option(help="batch size for training")] = 32,
    epochs: Annotated[int, typer.Option(help="number of epochs for training")] = 100,
):
    """
    Welcome!

    Use "--help" option to see usage information.
    """

    # ------------------ set up logger ------------------
    setUpLogger(level=level)

    logger.info(
        f"Launching....\nmodel: {model} dataset: {dataset} dataset_path: {dataset_path}"
        + f"device: {device} logging level: {level} task type: {task_type}"
        + f"split type: {split_type}"
    )

    split_dataset, num_subjects, num_electrodes, num_features, num_classes = load_data(
        dataset, dataset_path, sample_length, stride, task_type, split_type, label_type
    )

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
    typer.run(main)
