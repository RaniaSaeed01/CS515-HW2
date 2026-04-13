import os
import csv
import matplotlib.pyplot as plt
from typing import Dict, List


RESULTS_DIR = "results"


def load_test_accuracy(filepath: str) -> float:
    """
    Load overall test accuracy from a test CSV file.

    Args:
        filepath: Path to the test CSV file.

    Returns:
        Overall test accuracy as a float.
    """
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["class"] == "overall":
                return float(row["accuracy"])
    return 0.0


def load_train_csv(filepath: str) -> Dict[str, List]:
    """
    Load training history from a training CSV file.

    Args:
        filepath: Path to the training CSV file.

    Returns:
        Dictionary of lists for epoch, train_loss, val_loss,
        train_acc, val_acc.
    """
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in history:
                history[k].append(float(row[k]))
    return history


def plot_curves(
    histories:  Dict[str, Dict],
    metric:     str,
    title:      str,
    fname:      str,
    labels:     Dict[str, str],
) -> None:
    """
    Plot train/val metric curves for multiple runs.

    Args:
        histories: Dict of filename -> history dict.
        metric: 'loss' or 'acc'.
        title: Plot title.
        fname: Output filename.
        labels: Dict mapping filename to display label.
    """
    os.makedirs(f"{RESULTS_DIR}/summary", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    for key, history in histories.items():
        label = labels.get(key, key)
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
    labels:       Dict[str, str],
    baseline_acc: float = None,
) -> None:
    """
    Plot a bar chart of test accuracies.

    Args:
        results: Dict of filename -> test accuracy.
        title: Plot title.
        fname: Output filename.
        labels: Dict mapping filename to display label.
        baseline_acc: Optional baseline reference line.
    """
    os.makedirs(f"{RESULTS_DIR}/summary", exist_ok=True)
    display_labels = [labels.get(k, k) for k in results]
    values = list(results.values())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(display_labels, values, color="steelblue")
    if baseline_acc is not None:
        ax.axhline(baseline_acc, color="red", linestyle="--",
                   label=f"Baseline ({baseline_acc:.4f})")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Run")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(max(0, min(values) - 0.05), 1.0)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary/{fname}")
    plt.close()
    print(f"Saved: {RESULTS_DIR}/summary/{fname}")


def save_summary_csv(
    results: Dict[str, float],
    labels:  Dict[str, str],
    fname:   str,
) -> None:
    """
    Save a summary CSV of test accuracies.

    Args:
        results: Dict of filename -> test accuracy.
        labels: Dict mapping filename to display label.
        fname: Output filename.
    """
    os.makedirs(f"{RESULTS_DIR}/summary", exist_ok=True)
    fpath = f"{RESULTS_DIR}/summary/{fname}"
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    with open(fpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "label", "test_accuracy"])
        for key, acc in sorted_results:
            writer.writerow([key, labels.get(key, key), f"{acc:.4f}"])
    print(f"Saved: {fpath}")


def main() -> None:
    """
    Generate summary plots and tables for HW1b experiments.
    Covers transfer learning and knowledge distillation results.
    """

    # -------------------------------------------------------------------------
    # Transfer Learning — ResNet
    # -------------------------------------------------------------------------
    resnet_tl_files = {
        "resnet_pre=True_mode=finetune_freeze=True_ls=0.0_kd=False_T=4.0_alpha=0.7_st=full":
            "Finetune Frozen",
        "resnet_pre=True_mode=finetune_freeze=False_ls=0.0_kd=False_T=4.0_alpha=0.7_st=full":
            "Finetune Unfrozen",
        "resnet_pre=True_mode=scratch_freeze=False_ls=0.0_kd=False_T=4.0_alpha=0.7_st=full":
            "Scratch (modified conv)",
    }

    resnet_tl_results = {
        k: load_test_accuracy(f"{RESULTS_DIR}/{k}_test.csv")
        for k in resnet_tl_files
    }
    resnet_tl_histories = {
        k: load_train_csv(f"{RESULTS_DIR}/{k}.csv")
        for k in resnet_tl_files
    }

    plot_curves(resnet_tl_histories, "loss", "ResNet Transfer Learning: Loss",
                "resnet_tl_loss.png", resnet_tl_files)
    plot_curves(resnet_tl_histories, "acc",  "ResNet Transfer Learning: Accuracy",
                "resnet_tl_acc.png",  resnet_tl_files)
    plot_bar(resnet_tl_results, "ResNet Transfer Learning: Test Accuracy",
             "resnet_tl_bar.png", resnet_tl_files)

    # -------------------------------------------------------------------------
    # Transfer Learning — VGG
    # -------------------------------------------------------------------------
    vgg_tl_files = {
        "vgg_pre=True_mode=finetune_freeze=True_ls=0.0_kd=False_T=4.0_alpha=0.7_st=full":
            "Finetune Frozen",
        "vgg_pre=True_mode=finetune_freeze=False_ls=0.0_kd=False_T=4.0_alpha=0.7_st=full":
            "Finetune Unfrozen",
        "vgg_pre=True_mode=scratch_freeze=False_ls=0.0_kd=False_T=4.0_alpha=0.7_st=full":
            "Scratch (modified conv)",
    }

    vgg_tl_results = {
        k: load_test_accuracy(f"{RESULTS_DIR}/{k}_test.csv")
        for k in vgg_tl_files
    }
    vgg_tl_histories = {
        k: load_train_csv(f"{RESULTS_DIR}/{k}.csv")
        for k in vgg_tl_files
    }

    plot_curves(vgg_tl_histories, "loss", "VGG Transfer Learning: Loss",
                "vgg_tl_loss.png", vgg_tl_files)
    plot_curves(vgg_tl_histories, "acc",  "VGG Transfer Learning: Accuracy",
                "vgg_tl_acc.png",  vgg_tl_files)
    plot_bar(vgg_tl_results, "VGG Transfer Learning: Test Accuracy",
             "vgg_tl_bar.png", vgg_tl_files)

    # -------------------------------------------------------------------------
    # Label Smoothing
    # -------------------------------------------------------------------------
    ls_files = {
        "resnet_pre=False_mode=finetune_freeze=True_ls=0.0_kd=False_T=4.0_alpha=0.7_st=full":
            "ResNet (no LS)",
        "resnet_pre=False_mode=finetune_freeze=True_ls=0.1_kd=False_T=4.0_alpha=0.7_st=full":
            "ResNet (LS=0.1)",
    }

    ls_results = {
        k: load_test_accuracy(f"{RESULTS_DIR}/{k}_test.csv")
        for k in ls_files
    }
    ls_histories = {
        k: load_train_csv(f"{RESULTS_DIR}/{k}.csv")
        for k in ls_files
    }

    plot_curves(ls_histories, "loss", "Label Smoothing: Loss",
                "ls_loss.png", ls_files)
    plot_curves(ls_histories, "acc",  "Label Smoothing: Accuracy",
                "ls_acc.png",  ls_files)
    plot_bar(ls_results, "Label Smoothing: Test Accuracy",
             "ls_bar.png", ls_files)

    # -------------------------------------------------------------------------
    # Knowledge Distillation
    # -------------------------------------------------------------------------
    kd_files = {
        "cnn_pre=False_mode=finetune_freeze=True_ls=0.0_kd=False_T=4.0_alpha=0.7_st=full":
            "SimpleCNN (baseline)",
        "resnet_pre=False_mode=finetune_freeze=True_ls=0.0_kd=False_T=4.0_alpha=0.7_st=full":
            "ResNet (teacher)",
        "cnn_pre=False_mode=finetune_freeze=True_ls=0.0_kd=True_T=4.0_alpha=0.7_st=full":
            "SimpleCNN (KD)",
        "mobilenet_pre=False_mode=finetune_freeze=True_ls=0.0_kd=True_T=4.0_alpha=0.7_st=true_only":
            "MobileNet (true-only KD)",
    }

    kd_results = {
        k: load_test_accuracy(f"{RESULTS_DIR}/{k}_test.csv")
        for k in kd_files
    }
    kd_histories = {
        k: load_train_csv(f"{RESULTS_DIR}/{k}.csv")
        for k in kd_files
    }

    # Use ResNet teacher as baseline reference
    teacher_acc = kd_results[
        "resnet_pre=False_mode=finetune_freeze=True_ls=0.0_kd=False_T=4.0_alpha=0.7_st=full"
    ]

    plot_curves(kd_histories, "loss", "Knowledge Distillation: Loss",
                "kd_loss.png", kd_files)
    plot_curves(kd_histories, "acc",  "Knowledge Distillation: Accuracy",
                "kd_acc.png",  kd_files)
    plot_bar(kd_results, "Knowledge Distillation: Test Accuracy",
             "kd_bar.png", kd_files, baseline_acc=teacher_acc)

    # -------------------------------------------------------------------------
    # Full summary CSV
    # -------------------------------------------------------------------------
    all_files = {}
    all_files.update(resnet_tl_files)
    all_files.update(vgg_tl_files)
    all_files.update(ls_files)
    all_files.update(kd_files)

    all_results = {}
    for k in all_files:
        test_path = f"{RESULTS_DIR}/{k}_test.csv"
        if os.path.exists(test_path):
            all_results[k] = load_test_accuracy(test_path)

    save_summary_csv(all_results, all_files, "hw1b_all_results.csv")
    print("\nAll summary plots and tables saved to results/summary/")


if __name__ == "__main__":
    main()