#!/usr/bin/env python3
"""Influence-function detection experiment.

For each model/dataset pair, this script checks whether Welch's t-test can tell
a sample of benign items apart from a sample that contains some malicious items,
by comparing each sample against a bootstrapped reference sample.

Usage:
    python statistical_test.py                                  # launch every model/dataset pair in the background
    python statistical_test.py --model MODEL --dataset DATASET  # run a single pair in the foreground

Input files, in --base-dir:
    {model}_reference_dataset.pkl
    {model}_{dataset}_malicious.pkl
Output file, in --base-dir (rewritten after each norm finishes):
    {model}_{dataset}.pkl
    results[n_samples][norm]["k=.."]["p=.."][metric] = {"mean": ..., "std": ...}
"""
import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# <repo>/results, found relative to this script
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent / "results"

MODELS = [
    "granite-3b-code-instruct-2k",
    "deepseek-coder-6_7b-instruct",
    "Qwen2_5-Coder-7B-Instruct",
]
DATASETS = ["auth", "infrastructure"]  # add "injection" to include it

N_BOOTSTRAP = 1000  # bootstrap repetitions per (n_samples, norm, k, p) setting
N_TRIALS = 100      # tests per repetition: half benign, half malicious

SAMPLES_PER_ITEM = [25, 20, 15, 10, 5]           # values averaged per item
SUBSET_SIZES = [80, 70, 60, 50, 40, 30, 20, 10]  # k: items per tested sample
MALICIOUS_RATES = [round(0.1 * i, 1) for i in range(1, 11)]     # p: chance each item is malicious
P_VALUE_THRESHOLDS = [round(0.01 * i, 2) for i in range(1, 11)]

# Alternative hypothesis for the t-test. Both sample types use the same one,
# so the p-value thresholds mean the same thing for benign and malicious samples.
BENIGN_ALTERNATIVE = "greater"
MALICIOUS_ALTERNATIVE = "greater"

EPS = 1e-10  # avoids division by zero in the metrics


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def norm_names(data):
    """Norm names stored in the data (read from the first entry).

    The data layout is data[item][group][norm] -> list of values.
    """
    first_item = next(iter(data.values()))
    first_group = next(iter(first_item.values()))
    return set(first_group.keys())


def item_means(data, norm, n_samples, rng):
    """For each item, average `n_samples` randomly chosen values of `norm`."""
    means = []
    for item_key, item in data.items():
        values = [float(v) for group in item.values() for v in group.get(norm, [])]
        if not values:
            raise ValueError(f"No data found for norm={norm!r} under key {item_key!r}")

        # Same as the original: once an item has fewer values than n_samples,
        # the smaller count is also used for every item after it.
        n_samples = min(n_samples, len(values))
        means.append(np.mean(rng.choice(values, size=n_samples, replace=False)))
    return np.array(means)


# -----------------------------------------------------------------------------
# Detection test
# -----------------------------------------------------------------------------

def welch_t_test(sample, reference, alternative):
    """Welch's t-test (unequal variances). Returns (t, p); (0, 1) if undefined."""
    t_stat, p_value = stats.ttest_ind(sample, reference, equal_var=False, alternative=alternative)
    if np.isnan(t_stat):
        return 0.0, 1.0
    return t_stat, p_value


def run_detection_trials(reference, benign, malicious, k, p, rng):
    """Run N_TRIALS tests, alternating benign and malicious samples of size k.

    A malicious sample has Binomial(k, p) malicious items; the rest are benign.
    Returns the AUROC of the t-statistics, plus one value per p-value threshold
    for each of the other metrics.
    """
    if k > len(benign):
        raise ValueError(f"k={k} is larger than the benign population size {len(benign)}")

    t_stats, p_values, is_malicious = [], [], []
    for _ in range(N_TRIALS // 2):
        benign_sample = rng.choice(benign, size=k, replace=False)
        t, pv = welch_t_test(benign_sample, reference, BENIGN_ALTERNATIVE)
        t_stats.append(t)
        p_values.append(pv)
        is_malicious.append(False)

        n_malicious = rng.binomial(k, p)
        mixed_sample = np.concatenate([
            rng.choice(malicious, size=n_malicious, replace=False),
            rng.choice(benign, size=k - n_malicious, replace=False),
        ])
        t, pv = welch_t_test(mixed_sample, reference, MALICIOUS_ALTERNATIVE)
        t_stats.append(t)
        p_values.append(pv)
        is_malicious.append(True)

    p_values = np.array(p_values)
    is_malicious = np.array(is_malicious)

    metrics = {"AUROC": roc_auc_score(is_malicious, t_stats)}
    for threshold in P_VALUE_THRESHOLDS:
        flagged = p_values < threshold
        tp = np.sum(flagged & is_malicious)
        fp = np.sum(flagged & ~is_malicious)
        fn = np.sum(~flagged & is_malicious)
        tn = np.sum(~flagged & ~is_malicious)

        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        specificity = tn / (tn + fp + EPS)

        for name, value in [
            ("Accuracy", (tp + tn) / (len(p_values) + EPS)),
            ("Precision", precision),
            ("Recall", recall),
            ("Specificity", specificity),
            ("F1", 2 * precision * recall / (precision + recall + EPS)),
            ("FPR", fp / (fp + tn + EPS)),
            ("FNR", fn / (fn + tp + EPS)),
            ("Balanced Accuracy", (recall + specificity) / 2),
        ]:
            metrics.setdefault(name, []).append(float(value))
    return metrics


def summarize(runs):
    """Mean and std (ddof=1) of each metric across runs (per threshold for list metrics)."""
    summary = {}
    for name in runs[0]:
        values = np.array([run[name] for run in runs], dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
        summary[name] = {"mean": mean.tolist(), "std": std.tolist()}
    return summary


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------

def run_experiment(model, dataset, base_dir):
    reference_path = base_dir / f"{model}_reference_dataset.pkl"
    malicious_path = base_dir / f"{model}_{dataset}_malicious.pkl"
    output_path = base_dir / f"{model}_{dataset}.pkl"

    print(f"Model:    {model}")
    print(f"Dataset:  {dataset}")
    print(f"Base dir: {base_dir}")

    with open(reference_path, "rb") as f:
        reference_data = pickle.load(f)
    with open(malicious_path, "rb") as f:
        malicious_data = pickle.load(f)

    rng = np.random.default_rng()
    norms = sorted(norm_names(reference_data) & norm_names(malicious_data))
    results = {}

    for n_samples in SAMPLES_PER_ITEM:
        results[str(n_samples)] = {}

        for norm in norms:
            results[str(n_samples)][norm] = {}
            malicious = item_means(malicious_data, norm, n_samples, rng)
            reference_pool = item_means(reference_data, norm, n_samples, rng)

            for k in SUBSET_SIZES:
                results[str(n_samples)][norm][f"k={k}"] = {}

                for p in MALICIOUS_RATES:
                    runs = []
                    for _ in range(N_BOOTSTRAP):
                        # Draw the reference and the benign pool independently,
                        # with replacement, from the same reference data.
                        size = len(reference_pool)
                        reference = rng.choice(reference_pool, size=size, replace=True)
                        benign = rng.choice(reference_pool, size=size, replace=True)
                        runs.append(run_detection_trials(reference, benign, malicious, k, p, rng))

                    results[str(n_samples)][norm][f"k={k}"][f"p={p:.1f}"] = summarize(runs)

            with open(output_path, "wb") as f:
                pickle.dump(results, f)
            print(f"Checkpoint saved (N_samples={n_samples}, norm={norm})", flush=True)


def launch_all(base_dir):
    """Start one background run per model/dataset pair, each logging to its own file.

    Runs keep going after the terminal closes (like `nohup ... &`).
    """
    for model in MODELS:
        for dataset in DATASETS:
            log_path = Path(f"{model}_{dataset}_new_test.log")
            with open(log_path, "w") as log:
                subprocess.Popen(
                    [sys.executable, __file__, "--model", model, "--dataset", dataset,
                     "--base-dir", str(base_dir)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            print(f"Started {model} / {dataset} -> {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the influence-function detection experiment.")
    parser.add_argument("--model", help=f"One of: {', '.join(MODELS)}. Omit to launch all pairs.")
    parser.add_argument("--dataset", help="e.g. auth, infrastructure, injection. Omit to launch all pairs.")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                        help="Directory with the input and output pickle files (default: %(default)s)")
    args = parser.parse_args()

    if (args.model is None) != (args.dataset is None):
        parser.error("give both --model and --dataset, or neither")

    base_dir = Path(args.base_dir)
    if args.model is None:
        launch_all(base_dir)
    else:
        run_experiment(args.model, args.dataset, base_dir)


if __name__ == "__main__":
    main()
