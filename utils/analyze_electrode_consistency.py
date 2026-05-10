"""
Analyze consistency of per-electrode emotion classification accuracy across subjects.

Loads the per-subject electrode weight files and computes:
  - Per-electrode mean, std, and coefficient of variation across subjects
  - Pairwise Spearman rank correlations between all subject pairs
  - Top/bottom consistently high and low electrodes
  - Visualizations: heatmap, bar chart, correlation matrix

Usage:
    python utils/analyze_electrode_consistency.py --weights-dir ./cache/subjects
"""
import sys
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

app = typer.Typer(pretty_exceptions_show_locals=False)


def _load_weights(weights_dir: str) -> tuple[np.ndarray, list[str]]:
    """Load all electrode_weights_*.npy files, return (n_subjects, n_electrodes) array and subject labels."""
    p = Path(weights_dir)
    files = sorted(p.glob("electrode_weights_*.npy"))
    if not files:
        logger.error("No electrode_weights_*.npy files found in {}", weights_dir)
        raise SystemExit(1)
    weights = [np.load(str(f)) for f in files]
    subject_ids = [f.stem.split("_")[-1] for f in files]
    W = np.stack(weights, axis=0)
    logger.info("Loaded {} subjects, {} electrodes each", W.shape[0], W.shape[1])
    return W, subject_ids


def _print_table(mean_w, std_w, cv, rankings, n_top=10):
    """Print text summary of consistent high/low electrodes."""
    n_elec = len(mean_w)
    sorted_by_mean = np.argsort(mean_w)[::-1]  # highest first

    logger.info("\n=== Electrodes sorted by mean weight (highest first) ===")
    logger.info(f"{'Electrode':>9} | {'Mean':>8} | {'Std':>8} | {'CV':>7} | {'Rank range'}")
    logger.info("-" * 60)
    for e in sorted_by_mean:
        r_min = np.min(rankings[:, e]) + 1
        r_max = np.max(rankings[:, e]) + 1
        r_avg = np.mean(rankings[:, e]) + 1
        logger.info(f"    {e+1:>3}     | {mean_w[e]:>8.6f} | {std_w[e]:>8.6f} | {cv[e]:>7.3f} | [{r_min:.0f}-{r_max:.0f}] avg={r_avg:.1f}")

    # Top-10 consistently highest
    top_idx = sorted_by_mean[:n_top]
    logger.info("\n=== Top-{} consistently highest electrodes (by mean weight) ===", n_top)
    for e in top_idx:
        r_avg = np.mean(rankings[:, e]) + 1
        r_std = np.std(rankings[:, e])
        logger.info(f"  Electrode {e+1:>2}: mean={mean_w[e]:.6f}, avg_rank={r_avg:.1f} +/- {r_std:.1f}")

    # Bottom-10 consistently lowest
    bot_idx = sorted_by_mean[-n_top:]
    logger.info("\n=== Bottom-{} consistently lowest electrodes (by mean weight) ===", n_top)
    for e in bot_idx:
        r_avg = np.mean(rankings[:, e]) + 1
        r_std = np.std(rankings[:, e])
        logger.info(f"  Electrode {e+1:>2}: mean={mean_w[e]:.6f}, avg_rank={r_avg:.1f} +/- {r_std:.1f}")

    # Most variable (highest CV)
    most_var = np.argsort(cv)[::-1][:n_top]
    logger.info("\n=== Most variable electrodes (highest coefficient of variation) ===")
    for e in most_var:
        logger.info(f"  Electrode {e+1:>2}: CV={cv[e]:.3f}, mean={mean_w[e]:.6f}, std={std_w[e]:.6f}")


@app.command()
def main(
    weights_dir: str = typer.Option("./cache/subjects", help="Directory containing electrode_weights_*.npy files"),
    output_dir: str = typer.Option("./cache/subjects", help="Directory to save output figures"),
):
    W, subject_ids = _load_weights(weights_dir)
    n_sub, n_elec = W.shape
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Convert to log-space for better analysis ---
    log_W = np.log(W + 1e-10)

    # --- Per-electrode statistics ---
    mean_w = np.mean(W, axis=0)
    std_w = np.std(W, axis=0)
    cv = std_w / (mean_w + 1e-10)

    # Per-subject rankings (0 = lowest, 61 = highest)
    rankings = np.empty_like(W, dtype=int)
    for s in range(n_sub):
        rankings[s] = np.argsort(np.argsort(W[s]))

    # --- Pairwise Spearman rank correlations ---
    rhos = np.zeros((n_sub, n_sub))
    for i in range(n_sub):
        for j in range(n_sub):
            if i == j:
                rhos[i, j] = 1.0
            elif i < j:
                rho, _ = spearmanr(W[i], W[j])
                rhos[i, j] = rho
                rhos[j, i] = rho

    off_diag = rhos[np.triu_indices(n_sub, k=1)]
    logger.info("\n=== Overall Consistency ===")
    logger.info(f"Mean pairwise Spearman rho: {off_diag.mean():.4f} +/- {off_diag.std():.4f}")
    logger.info(f"Min pairwise rho: {off_diag.min():.4f}")
    logger.info(f"Max pairwise rho: {off_diag.max():.4f}")

    if off_diag.mean() > 0.7:
        logger.info("=> HIGH consistency: electrode importance is largely consistent across subjects")
    elif off_diag.mean() > 0.4:
        logger.info("=> MODERATE consistency: some electrodes are consistently important, but patterns vary")
    else:
        logger.info("=> LOW consistency: electrode importance appears mostly random across subjects")

    # --- Print text table ---
    _print_table(mean_w, std_w, cv, rankings)

    # --- Plot 1: Heatmap (subjects x electrodes) - softmax weights ---
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(W, aspect="auto", cmap="viridis")
    ax.set_xticks(range(n_elec))
    ax.set_xticklabels([str(e + 1) for e in range(n_elec)], fontsize=6)
    ax.set_yticks(range(n_sub))
    ax.set_yticklabels([f"S{s}" for s in subject_ids])
    ax.set_xlabel("Electrode Index (1-based)")
    ax.set_ylabel("Subject")
    ax.set_title("Electrode Weights (softmax-normalized) across Subjects")
    fig.colorbar(im, ax=ax, label="Weight")
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "heatmap.png"), dpi=150)
    plt.close(fig)
    logger.info("\nSaved weight heatmap to {}/heatmap.png", output_dir)

    # --- Plot 1b: Heatmap based on recovered relative accuracy ---
    # log(w_i) - mean(log(w)) = acc_i - mean(acc) within each subject.
    # This removes the per-subject softmax constant shift, revealing true
    # per-electrode accuracy differences.
    log_W = np.log(W + 1e-10)
    acc_relative = log_W - log_W.mean(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(acc_relative, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(n_elec))
    ax.set_xticklabels([str(e + 1) for e in range(n_elec)], fontsize=6)
    ax.set_yticks(range(n_sub))
    ax.set_yticklabels([f"S{s}" for s in subject_ids])
    ax.set_xlabel("Electrode Index (1-based)")
    ax.set_ylabel("Subject")
    ax.set_title("Per-Electrode Relative Accuracy (recovered from softmax weights)")
    fig.colorbar(im, ax=ax, label="Relative accuracy (acc - mean acc)")
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "heatmap_acc.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy heatmap to {}/heatmap_acc.png", output_dir)

    # --- Plot 2: Mean +/- Std bar chart ---
    sorted_idx = np.argsort(mean_w)
    fig, ax = plt.subplots(figsize=(18, 5))
    x = np.arange(n_elec)
    ax.bar(x, mean_w[sorted_idx], yerr=std_w[sorted_idx], capsize=2, color="steelblue", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([str(sorted_idx[e] + 1) for e in range(n_elec)], fontsize=6, rotation=90)
    ax.set_xlabel("Electrode Index (1-based), sorted by mean")
    ax.set_ylabel("Mean Weight")
    ax.set_title("Per-Electrode Mean Weight +/- Std Dev Across Subjects")
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "mean_std_bar.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved bar chart to {}/mean_std_bar.png", output_dir)

    # --- Plot 3: Subject correlation matrix ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(rhos, vmin=0, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n_sub))
    ax.set_yticks(range(n_sub))
    ax.set_xticklabels([f"S{s}" for s in subject_ids], fontsize=8)
    ax.set_yticklabels([f"S{s}" for s in subject_ids], fontsize=8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Subject")
    ax.set_title("Pairwise Spearman Rank Correlation of Electrode Weights")
    for i in range(n_sub):
        for j in range(n_sub):
            ax.text(j, i, f"{rhos[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if rhos[i, j] > 0.7 else "black")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "correlation_matrix.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved correlation matrix to {}/correlation_matrix.png", output_dir)


if __name__ == "__main__":
    app()
