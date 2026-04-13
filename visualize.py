import os
import csv
import matplotlib.pyplot as plt
from typing import Dict, List


RESULTS_DIR = "results"


def load_test_csvs(results_dir: str) -> Dict[str, dict]:
    """
    Load all test CSV files from the results directory.

    Args:
        results_dir: Path to the results folder.

    Returns:
        Dictionary mapping run name to its overall test accuracy.
    """
    results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_test.csv"):
            run_name = fname.replace("_test.csv", "")
            fpath = os.path.join(results_dir, fname)
            with open(fpath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["class"] == "overall":
                        results[run_name] = float(row["accuracy"])
    return results


def load_train_csv(results_dir: str, run_name: str) -> Dict[str, List]:
    """
    Load a single training CSV file.

    Args:
        results_dir: Path to the results folder.
        run_name: The run identifier string.

    Returns:
        Dictionary of lists for epoch, train_loss, val_loss,
        train_acc, val_acc.
    """
    fpath = os.path.join(results_dir, f"{run_name}.csv")
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    with open(fpath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in history:
                history[k].append(float(row[k]))
    return history


def filter_runs(all_results: Dict[str, float], keyword: str) -> Dict[str, float]:
    """
    Filter runs by a keyword present in the run name.

    Args:
        all_results: Full dictionary of run_name -> accuracy.
        keyword: Substring to filter by.

    Returns:
        Filtered dictionary of matching runs.
    """
    return {k: v for k, v in all_results.items() if keyword in k}

def extract_label(run_name: str, label_key: str) -> str:
    """
    Extract the value of a specific hyperparameter from a run name string.

    Args:
        run_name: Full run name string e.g. 'hidden=512-256-128_act=relu_drop=0.3...'
        label_key: The key to extract e.g. 'drop', 'act', 'bn', 'hidden'.

    Returns:
        The value string for that key, or the full run name if not found.
    """
    for part in run_name.split("_"):
        if part.startswith(label_key + "="):
            return part
    return run_name

def plot_comparison(
    histories:  Dict[str, Dict],
    metric:     str,
    title:      str,
    fname:      str,
    label_key:  str = "",
) -> None:
    """
    Plot a train/val metric across multiple runs on the same axes.

    Args:
        histories: Dict of run_name -> history dict.
        metric: One of 'loss' or 'acc'.
        title: Plot title.
        fname: Output filename (saved to results/summary/).
        label_key: The hyperparameter key to extract for the legend label.
    """
    os.makedirs(f"{RESULTS_DIR}/summary", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    for run_name, history in histories.items():
        # Extract just the relevant hyperparameter value for the legend
        label = extract_label(run_name, label_key)
        ax1.plot(history["epoch"], history[f"train_{metric}"], label=label)
        ax2.plot(history["epoch"], history[f"val_{metric}"],   label=label)

    ax1.set_title(f"Train {metric.capitalize()}")
    ax2.set_title(f"Val {metric.capitalize()}")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary/{fname}")
    plt.close()
    print(f"Saved: {RESULTS_DIR}/summary/{fname}")


def plot_bar(
    results:      Dict[str, float],
    title:        str,
    fname:        str,
    baseline_acc: float,
    label_key:    str = "",
) -> None:
    """
    Plot a bar chart of test accuracies for a group of runs.

    Args:
        results: Dict of run_name -> test accuracy.
        title: Plot title.
        fname: Output filename (saved to results/summary/).
        baseline_acc: Baseline accuracy drawn as a reference line.
        label_key: The hyperparameter key to extract for the x-axis labels.
    """
    os.makedirs(f"{RESULTS_DIR}/summary", exist_ok=True)

    labels = [extract_label(k, label_key) for k in results]
    values = list(results.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="steelblue")
    ax.axhline(baseline_acc, color="red", linestyle="--",
               label=f"Baseline ({baseline_acc:.4f})")
    ax.set_title(title)
    ax.set_xlabel("Run")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(min(values) - 0.01, 1.0)
    ax.legend()
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary/{fname}")
    plt.close()
    print(f"Saved: {RESULTS_DIR}/summary/{fname}")


def save_summary_table(all_results: Dict[str, float], baseline_acc: float) -> None:
    """
    Save a summary CSV table of all runs sorted by test accuracy.

    Args:
        all_results: Full dictionary of run_name -> accuracy.
        baseline_acc: Baseline accuracy for delta calculation.
    """
    os.makedirs(f"{RESULTS_DIR}/summary", exist_ok=True)
    fpath = f"{RESULTS_DIR}/summary/all_results.csv"

    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

    with open(fpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "test_accuracy", "delta_vs_baseline"])
        for run_name, acc in sorted_results:
            delta = acc - baseline_acc
            writer.writerow([run_name, f"{acc:.4f}", f"{delta:+.4f}"])

    print(f"Saved summary table: {fpath}")

def get_param_value(run_name: str, key: str) -> str:
    """
    Extract the exact value of a hyperparameter from a run name.

    Args:
        run_name: Full run name string.
        key: Hyperparameter key e.g. 'wd', 'drop', 'bn'.

    Returns:
        The exact value string, or empty string if not found.
    """
    for part in run_name.split("_"):
        if part.startswith(key + "="):
            return part.split("=")[1]
    return ""


def match_params(run_name: str, conditions: dict) -> bool:
    """
    Check if a run name exactly matches all given hyperparameter conditions.

    Args:
        run_name: Full run name string.
        conditions: Dict of key -> expected value e.g. {'wd': '0.0001', 'drop': '0.3'}.

    Returns:
        True if all conditions match exactly, False otherwise.
    """
    for key, val in conditions.items():
        if get_param_value(run_name, key) != val:
            return False
    return True


def main() -> None:
    """
    Load all experiment results and generate summary plots and tables
    for each ablation group using exact hyperparameter matching.
    """
    all_results = load_test_csvs(RESULTS_DIR)

    # Fixed baseline hyperparameters
    baseline_conditions = {
        "hidden": "512-256-128",
        "act":    "relu",
        "drop":   "0.3",
        "bn":     "True",
        "wd":     "0.0001",
        "l1":     "0.0",
        "sched":  "step",
    }

    baseline_key = [k for k in all_results if match_params(k, baseline_conditions)]
    if not baseline_key:
        print("Baseline run not found.")
        return

    baseline_acc = all_results[baseline_key[0]]
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # --- Architecture: vary hidden only ---
    arch_fixed = {k: v for k in baseline_conditions if k != "hidden"
                  for v in [baseline_conditions[k]]}
    arch_runs = {k: v for k, v in all_results.items()
                 if match_params(k, {p: baseline_conditions[p]
                                     for p in baseline_conditions if p != "hidden"})}
    arch_histories = {k: load_train_csv(RESULTS_DIR, k) for k in arch_runs}
    plot_comparison(arch_histories, "loss", "Architecture: Loss Curves",     "arch_loss.png", label_key="hidden")
    plot_comparison(arch_histories, "acc",  "Architecture: Accuracy Curves", "arch_acc.png",  label_key="hidden")
    plot_bar(arch_runs, "Architecture: Test Accuracy", "arch_bar.png", baseline_acc, label_key="hidden")

    # --- Activation: vary act only ---
    act_runs = {k: v for k, v in all_results.items()
                if match_params(k, {p: baseline_conditions[p]
                                    for p in baseline_conditions if p != "act"})}
    act_histories = {k: load_train_csv(RESULTS_DIR, k) for k in act_runs}
    plot_comparison(act_histories, "loss", "Activation: Loss Curves",     "act_loss.png", label_key="act")
    plot_comparison(act_histories, "acc",  "Activation: Accuracy Curves", "act_acc.png",  label_key="act")
    plot_bar(act_runs, "Activation: Test Accuracy", "act_bar.png", baseline_acc, label_key="act")

    # --- Dropout: vary drop only ---
    drop_runs = {k: v for k, v in all_results.items()
                 if match_params(k, {p: baseline_conditions[p]
                                     for p in baseline_conditions if p != "drop"})}
    drop_histories = {k: load_train_csv(RESULTS_DIR, k) for k in drop_runs}
    plot_comparison(drop_histories, "loss", "Dropout: Loss Curves",     "drop_loss.png", label_key="drop")
    plot_comparison(drop_histories, "acc",  "Dropout: Accuracy Curves", "drop_acc.png",  label_key="drop")
    plot_bar(drop_runs, "Dropout: Test Accuracy", "drop_bar.png", baseline_acc, label_key="drop")

    # --- BatchNorm: vary bn only ---
    bn_runs = {k: v for k, v in all_results.items()
               if match_params(k, {p: baseline_conditions[p]
                                   for p in baseline_conditions if p != "bn"})}
    bn_histories = {k: load_train_csv(RESULTS_DIR, k) for k in bn_runs}
    plot_comparison(bn_histories, "loss", "BatchNorm: Loss Curves",     "bn_loss.png", label_key="bn")
    plot_comparison(bn_histories, "acc",  "BatchNorm: Accuracy Curves", "bn_acc.png",  label_key="bn")
    plot_bar(bn_runs, "BatchNorm: Test Accuracy", "bn_bar.png", baseline_acc, label_key="bn")

    # --- Regularization: vary wd and l1 ---
    reg_runs = {k: v for k, v in all_results.items()
                if match_params(k, {p: baseline_conditions[p]
                                    for p in baseline_conditions if p not in ("wd", "l1")})}
    reg_histories = {k: load_train_csv(RESULTS_DIR, k) for k in reg_runs}
    plot_comparison(reg_histories, "loss", "Regularization: Loss Curves",     "reg_loss.png", label_key="wd")
    plot_comparison(reg_histories, "acc",  "Regularization: Accuracy Curves", "reg_acc.png",  label_key="wd")
    plot_bar(reg_runs, "Regularization: Test Accuracy", "reg_bar.png", baseline_acc, label_key="wd")

    # --- Scheduler: vary sched only ---
    sched_runs = {k: v for k, v in all_results.items()
                  if match_params(k, {p: baseline_conditions[p]
                                      for p in baseline_conditions if p != "sched"})}
    sched_histories = {k: load_train_csv(RESULTS_DIR, k) for k in sched_runs}
    plot_comparison(sched_histories, "loss", "Scheduler: Loss Curves",     "sched_loss.png", label_key="sched")
    plot_comparison(sched_histories, "acc",  "Scheduler: Accuracy Curves", "sched_acc.png",  label_key="sched")
    plot_bar(sched_runs, "Scheduler: Test Accuracy", "sched_bar.png", baseline_acc, label_key="sched")

    # --- Summary table ---
    save_summary_table(all_results, baseline_acc)


if __name__ == "__main__":
    main()