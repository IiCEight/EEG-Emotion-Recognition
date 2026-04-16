"""
Compute per-electrode emotion classification accuracy and save softmax weights.

Usage:
    python utils/compute_electrode_weights.py --dataset-path ../data/SEED --cache-dir ./cache

Output:
    cache/electrode_weights.npy  — shape (62,) softmax-normalized accuracy weights
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataloder import load_data
from data.utils import merge_for_all_subjects


app = typer.Typer(pretty_exceptions_show_locals=False)


class ElectrodeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.net(x)


def train_electrode_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
) -> ElectrodeMLP:
    model = ElectrodeMLP()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    return model


def evaluate_accuracy(model: ElectrodeMLP, X_test: np.ndarray, y_test: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X).argmax(dim=1).numpy()
    return float((preds == y_test).mean())


@app.command()
def main(
    dataset_path: str = typer.Option(..., help="Path to SEED dataset"),
    cache_dir: str = typer.Option("./cache", help="Cache directory for loaded dataset and output"),
    epochs: int = typer.Option(200, help="Training epochs per electrode MLP"),
    seed: int = typer.Option(42, help="Random seed"),
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("Loading SEED dataset from {}", dataset_path)
    data, labels, num_subjects, num_electrodes, num_features, num_classes = load_data(
        dataset_name="SEED",
        dataset_path=dataset_path,
        cache_dir=cache_dir,
    )

    # Pool all sessions × all subjects
    all_data = []
    all_labels = []
    for session_id in range(len(data)):
        session_data = [data[session_id][s] for s in range(len(data[session_id]))]
        session_labels = [labels[session_id][s] for s in range(len(labels[session_id]))]
        d, l = merge_for_all_subjects(session_data, session_labels)
        all_data.append(d)
        all_labels.append(l)

    # Shape: (N, 62, 5) and (N,)
    X = np.concatenate(all_data, axis=0)   # (N, electrode, feature)
    y = np.concatenate(all_labels, axis=0) # (N,)
    logger.info("Pooled data shape: {}, labels shape: {}", X.shape, y.shape)

    accuracies = np.zeros(num_electrodes)
    for e in range(num_electrodes):
        X_e = X[:, e, :]  # (N, 5)
        X_train, X_test, y_train, y_test = train_test_split(
            X_e, y, test_size=0.2, random_state=seed, stratify=y
        )
        model = train_electrode_mlp(X_train, y_train, epochs=epochs)
        acc = evaluate_accuracy(model, X_test, y_test)
        accuracies[e] = acc
        if (e + 1) % 10 == 0:
            logger.info("Electrode {}/{} — acc: {:.4f}", e + 1, num_electrodes, acc)

    logger.info("Accuracy range: {:.4f} – {:.4f}", accuracies.min(), accuracies.max())

    # Softmax to get weights
    exp_acc = np.exp(accuracies - accuracies.max())  # numerically stable softmax
    weights = exp_acc / exp_acc.sum()

    out_path = Path(cache_dir) / "electrode_weights.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), weights)
    logger.info("Saved electrode weights to {}", out_path)
    logger.info("Weight range: {:.6f} – {:.6f}", weights.min(), weights.max())


if __name__ == "__main__":
    app()
