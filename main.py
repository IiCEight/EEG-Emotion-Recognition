from typing import Annotated
from config.logging import setUpLogger

import typer
import constant.CLI_arguments_enum as cli_enum
from loguru import logger

from data.dataloder import get_data


def main(
    model: Annotated[
        cli_enum.ModelName, typer.Argument(help="model name")
    ] = cli_enum.ModelName.DGCNN,
    dataset: Annotated[
        cli_enum.DatasetName, typer.Argument(help="dataset name")
    ] = cli_enum.DatasetName.DEAP,
    dataset_path: Annotated[
        str, typer.Argument(help="path to the dataset")
    ] = "../LibEER/data/DEAP",
    device: Annotated[str, typer.Argument(help="device to run the model on")] = "cpu",
    level: Annotated[
        cli_enum.LevelName, typer.Argument(help="level of severity for logging")
    ] = cli_enum.LevelName.DEBUG,
    sample_length: Annotated[int, typer.Option(help="length of data points in each sample")] = 128,
    stride: Annotated[int, typer.Option(help="stride for segmenting data")] = 128,
    label_type: Annotated[str, typer.Option(help="type of label to use (valence, arousal)")] = "valence",
):
    """
    Welcome!

    Use "--help" option to see usage information.
    """

    # ------------------ set up logger ------------------
    setUpLogger(level=level)

    logger.info(
        f"Launching....\nmodel: {model} dataset: {dataset} dataset_path: "
        + "{dataset_path} device: {device} logging level: {level}"
    )


    data, labels = get_data(dataset, dataset_path, sample_length, stride, label_type)


if __name__ == "__main__":
    typer.run(main)
