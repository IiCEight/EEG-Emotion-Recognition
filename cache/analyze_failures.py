"""
Verify and visualize failure sample distribution across two log files.
Checks whether failures are front-loaded (start at sample 0 of trials).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FILES = {
    "7s window": "failure_log_7s_window.csv",
    "4s length": "faiure_log_4s_length.csv",
}

def analyze(name, path):
    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"File: {path}  ({name})")
    print(f"  Total failure rows : {len(df)}")
    print(f"  Subjects           : {sorted(df['subject'].unique())}")
    print(f"  Sessions           : {sorted(df['session'].unique())}")

    # Per-trial stats
    trial_stats = (
        df.groupby(["subject", "session", "trial"])
        .agg(
            first_sample=("sample_in_trial", "min"),
            first_pct=("position_pct", "min"),
            last_pct=("position_pct", "max"),
            fail_count=("sample_in_trial", "count"),
            trial_length=("trial_length", "first"),
        )
        .reset_index()
    )
    trial_stats["fail_rate"] = trial_stats["fail_count"] / trial_stats["trial_length"]
    trial_stats["span_pct"] = trial_stats["last_pct"] - trial_stats["first_pct"]

    n_trials = len(trial_stats)
    at_zero = (trial_stats["first_sample"] == 0).sum()

    # "Whole-trial" failures: fail_rate > 50% — model is wrong for most of the trial
    mostly_wrong = (trial_stats["fail_rate"] > 0.5).sum()
    # Clustered: failures span < 20% of trial (tight cluster, not scattered)
    tight_cluster = (trial_stats["span_pct"] < 20).sum()
    # Scattered: span_pct > 60%
    scattered = (trial_stats["span_pct"] > 60).sum()

    print(f"\n  Failing trials total           : {n_trials}")
    print(f"  First failure at sample 0      : {at_zero}  ({at_zero/n_trials*100:.1f}%)")
    print(f"  First failure NOT at sample 0  : {n_trials - at_zero}  ({(n_trials-at_zero)/n_trials*100:.1f}%)")
    print(f"\n  Clustering analysis (per failing trial):")
    print(f"  Mostly wrong (fail_rate>50%)   : {mostly_wrong}  ({mostly_wrong/n_trials*100:.1f}%)")
    print(f"  Tight cluster (span<20%)       : {tight_cluster}  ({tight_cluster/n_trials*100:.1f}%)")
    print(f"  Scattered (span>60%)           : {scattered}  ({scattered/n_trials*100:.1f}%)")
    print(f"\n  fail_rate percentiles (failures / trial_length):")
    for p in [0, 25, 50, 75, 90, 100]:
        print(f"    p{p:3d}: {np.percentile(trial_stats['fail_rate'], p):.2f}")
    print(f"\n  span_pct percentiles (last_fail% - first_fail%):")
    for p in [0, 25, 50, 75, 90, 100]:
        print(f"    p{p:3d}: {np.percentile(trial_stats['span_pct'], p):.1f}%")

    return df, trial_stats


def plot(results):
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Failure Sample Clustering Analysis", fontsize=14)
    bins = np.arange(0, 105, 5)

    for col, (name, (df, ts)) in enumerate(results.items()):
        # Row 0: all failure positions
        axes[0, col].hist(df["position_pct"], bins=bins, color="steelblue", edgecolor="white")
        axes[0, col].set_title(f"[{name}] All failure positions (n={len(df)})")
        axes[0, col].set_xlabel("Position in trial (%)")
        axes[0, col].set_ylabel("Failure count")
        axes[0, col].axvline(10, color="red", linestyle="--", linewidth=1)

        # Row 1: fail_rate per trial (how much of the trial is wrong)
        axes[1, col].hist(ts["fail_rate"], bins=np.arange(0, 1.05, 0.05),
                          color="darkorange", edgecolor="white")
        mostly = (ts["fail_rate"] > 0.5).sum()
        axes[1, col].axvline(0.5, color="red", linestyle="--", linewidth=1)
        axes[1, col].set_title(
            f"[{name}] Fail rate per trial\n>50% wrong: {mostly}/{len(ts)} ({mostly/len(ts)*100:.0f}%)"
        )
        axes[1, col].set_xlabel("Fraction of trial samples that are wrong")
        axes[1, col].set_ylabel("Trials")

        # Row 2: span (last_fail% - first_fail%) — tight cluster vs. scattered
        axes[2, col].hist(ts["span_pct"], bins=bins, color="mediumpurple", edgecolor="white")
        tight = (ts["span_pct"] < 20).sum()
        axes[2, col].axvline(20, color="red", linestyle="--", linewidth=1, label="20% span")
        axes[2, col].set_title(
            f"[{name}] Failure span per trial\nspan<20%: {tight}/{len(ts)} ({tight/len(ts)*100:.0f}%)"
        )
        axes[2, col].set_xlabel("Span of failures in trial (last% − first%)")
        axes[2, col].set_ylabel("Trials")
        axes[2, col].legend(fontsize=8)

    plt.tight_layout()
    out = "failure_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    results = {}
    for name, fname in FILES.items():
        df, trial_stats = analyze(name, fname)
        results[name] = (df, trial_stats)

    plot(results)
