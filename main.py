# Used to parse command line arguments
from enum import Enum
from typing import Annotated
import typer


# Define an enumeration for model names.
# This helps in restricting the input arguments to specific model names
class ModelName(str, Enum):
    dgcnn = "DGCNN"
    DANN = "DANN"

class DatasetName(str, Enum):
    deap = "deap"
    seed = "seed"

def main(
    model:          Annotated[ModelName, typer.Argument(help="model name")] 
                        = ModelName.dgcnn,
    dataset:        Annotated[DatasetName, typer.Argument(help="dataset name")] 
                        = DatasetName.deap,
    dataset_path:   Annotated[str, typer.Argument(help="path to the dataset")] 
                        = "./data",
    device:         Annotated[str, 
                        typer.Argument(help="device to run the model on")] 
                        = "cpu",
):
    """
    Welcome!
    Use "--help" option to see usage information.
    """


    print("Hello from eer!")


if __name__ == "__main__":
    typer.run(main)
