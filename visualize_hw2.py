import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


RESULTS_DIR = "results"
CORRUPTIONS = ["fog", "brightness", "contrast", "gaussian_noise", "shot_noise"]
SEVERITIES  = [1, 3, 5]


def load_corrupted_results(prefix: str) -> Dict[str, Dict[int, float]]:
    """
    Load all corrupted test results for a given model prefix.

    Args:
        prefix: Run name prefix to match files against.

    Returns:
        Dict mapping corruption type to dict of severity -> accuracy.
    """
    results = {c: {} for c in CORRUPTIONS}
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            fname = f"{prefix}_corrupted_{corruption}_s{severity}.csv"
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        results[corruption][severity] = float(row["accuracy"])
    return results


def load_adversarial_results(prefix: str) -> Dict[str, float]:
    """
    Load adversarial test results for a given model prefix.

    Args:
        prefix: Run name prefix to match files against.

    Returns:
        Dict mapping attack type to accuracy.
    """
    fpath = os.path.join(RESULTS_DIR, f"{prefix}_adversarial.csv")
    results = {}
    if os.path.exists(fpath):
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[row["attack"]] = float(row["accuracy"])
    return results


def load_transferability_results(prefix: str) -> Dict[str, float]:
    """
    Load transferability test results for a given model prefix.

    Args:
        prefix: Run name prefix to match files against.

    Returns:
        Dict mapping model role to accuracy.
    """
    fpath = os.path.join(RESULTS_DIR, f"{prefix}_transferability.csv")
    results = {}
    if os.path.exists(fpath):
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[row["model"]] = float(row["accuracy"])
    return results


def plot_corrupted_comparison(
    standard_results: Dict,
    augmix_results:   Dict,
    clean_standard:   float,
    clean_augmix:     float,
) -> None:
    """
    Plot corrupted accuracy comparison between standard and AugMix models.

    Args:
        standard_results: Corrupted results for standard fine-tuned ResNet.
        augmix_results: Corrupted results for AugMix ResNet.
        clean_standard: Clean test accuracy of standard model.
        clean_augmix: Clean test accuracy of AugMix model.
    """
    os.makedirs(f"{RESULTS_DIR}/summary", exist_ok=True)

    # Plot per corruption type across severities
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Corrupted Test Accuracy: Standard vs AugMix ResNet", fontsize=13)
    axes = axes.flatten()

    for idx, corruption in enumerate(CORRUPTIONS):
        ax = axes[idx]
        std_accs = [standard_results[corruption].get(s, 0) for s in SEVERITIES]
        aug_accs = [augmix_results[corruption].get(s, 0)   for s in SEVERITIES]

        ax.plot(SEVERITIES, std_accs, "b-o", label="Standard")
        ax.plot(SEVERITIES, aug_accs, "r-o", label="AugMix")
        ax.axhline(clean_standard, color="b", linestyle="--", alpha=0.4)
        ax.axhline(clean_augmix,   color="r", linestyle="--", alpha=0.4)
        ax.set_title(corruption.replace("_", " ").title())
        ax.set_xlabel("Severity")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(SEVERITIES)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)

    # Hide unused subplot
    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary/corrupted_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/summary/corrupted_comparison.png")

    # Plot average accuracy across all corruptions per severity
    fig, ax = plt.subplots(figsize=(8, 5))
    for severity in SEVERITIES:
        std_avg = np.mean([standard_results[c].get(severity, 0) for c in CORRUPTIONS])
        aug_avg = np.mean([augmix_results[c].get(severity, 0)   for c in CORRUPTIONS])

    std_avgs = [np.mean([standard_results[c].get(s, 0) for c in CORRUPTIONS])
                for s in SEVERITIES]
    aug_avgs = [np.mean([augmix_results[c].get(s, 0) for c in CORRUPTIONS])
                for s in SEVERITIES]

    ax.plot(SEVERITIES, std_avgs, "b-o", label=f"Standard (clean={clean_standard:.3f})")
    ax.plot(SEVERITIES, aug_avgs, "r-o", label=f"AugMix (clean={clean_augmix:.3f})")
    ax.axhline(clean_standard, color="b", linestyle="--", alpha=0.4)
    ax.axhline(clean_augmix,   color="r", linestyle="--", alpha=0.4)
    ax.set_title("Average Corrupted Accuracy Across All Corruptions")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Average Accuracy")
    ax.set_xticks(SEVERITIES)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary/corrupted_average.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/summary/corrupted_average.png")


def plot_adversarial_comparison(
    standard_adv: Dict,
    augmix_adv:   Dict,
    clean_standard: float,
    clean_augmix:   float,
) -> None:
    """
    Plot adversarial robustness comparison between standard and AugMix models.

    Args:
        standard_adv: Adversarial results for standard ResNet.
        augmix_adv: Adversarial results for AugMix ResNet.
        clean_standard: Clean accuracy of standard model.
        clean_augmix: Clean accuracy of AugMix model.
    """
    os.makedirs(f"{RESULTS_DIR}/summary", exist_ok=True)

    attacks = ["pgd_linf", "pgd_l2"]
    x       = np.arange(len(attacks))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    std_vals = [standard_adv.get(a, 0) for a in attacks]
    aug_vals = [augmix_adv.get(a, 0)   for a in attacks]

    bars1 = ax.bar(x - width/2, std_vals, width, label="Standard", color="steelblue")
    bars2 = ax.bar(x + width/2, aug_vals, width, label="AugMix",   color="tomato")

    ax.axhline(clean_standard, color="steelblue", linestyle="--",
               alpha=0.5, label=f"Standard clean ({clean_standard:.3f})")
    ax.axhline(clean_augmix,   color="tomato",    linestyle="--",
               alpha=0.5, label=f"AugMix clean ({clean_augmix:.3f})")

    ax.set_title("Adversarial Robustness: Standard vs AugMix ResNet")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(["PGD-20 L-inf\n(eps=4/255)", "PGD-20 L2\n(eps=0.25)"])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary/adversarial_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/summary/adversarial_comparison.png")


def plot_transferability(transfer_results: Dict) -> None:
    """
    Plot adversarial transferability results.

    Args:
        transfer_results: Dict mapping model role to accuracy.
    """
    os.makedirs(f"{RESULTS_DIR}/summary", exist_ok=True)

    labels = ["Teacher on\nteacher adv", "Student on\nteacher adv"]
    values = [
        transfer_results.get("teacher_on_teacher_adv", 0),
        transfer_results.get("student_on_teacher_adv", 0),
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=["steelblue", "tomato"])
    ax.set_title("Adversarial Transferability\n(PGD-20 L-inf generated from AugMix Teacher)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary/transferability.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/summary/transferability.png")


def save_summary_csv(
    standard_results: Dict,
    augmix_results:   Dict,
    standard_adv:     Dict,
    augmix_adv:       Dict,
    transfer_results: Dict,
    clean_standard:   float,
    clean_augmix:     float,
) -> None:
    """Save a full summary CSV of all HW2 results."""
    os.makedirs(f"{RESULTS_DIR}/summary", exist_ok=True)
    fpath = f"{RESULTS_DIR}/summary/hw2_all_results.csv"

    with open(fpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "model", "condition", "accuracy"])

        # Clean accuracy
        writer.writerow(["clean", "standard", "clean", f"{clean_standard:.4f}"])
        writer.writerow(["clean", "augmix",   "clean", f"{clean_augmix:.4f}"])

        # Corrupted
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                std_acc = standard_results[corruption].get(severity, "N/A")
                aug_acc = augmix_results[corruption].get(severity, "N/A")
                writer.writerow(["corrupted", "standard",
                                 f"{corruption}_s{severity}", std_acc])
                writer.writerow(["corrupted", "augmix",
                                 f"{corruption}_s{severity}", aug_acc])

        # Adversarial
        for attack, acc in standard_adv.items():
            writer.writerow(["adversarial", "standard", attack, f"{acc:.4f}"])
        for attack, acc in augmix_adv.items():
            writer.writerow(["adversarial", "augmix", attack, f"{acc:.4f}"])

        # Transferability
        for role, acc in transfer_results.items():
            writer.writerow(["transferability", role, "pgd_linf", f"{acc:.4f}"])

    print(f"Saved: {fpath}")


def main() -> None:
    """Generate all HW2 summary plots and tables."""

    # Model prefixes — using scratch-trained models for corrupted evaluation
    # since finetune models expect 224x224 input but CIFAR-10-C is 32x32
    standard_prefix = ("resnet_pre=False_mode=finetune_freeze=True_ls=0.0"
                       "_kd=False_augmix=False_T=4.0_alpha=0.7_st=full")
    augmix_prefix   = ("resnet_pre=False_mode=finetune_freeze=True_ls=0.0"
                       "_kd=False_augmix=True_T=4.0_alpha=0.7_st=full")
    adv_standard_prefix = ("resnet_pre=True_mode=finetune_freeze=False_ls=0.0"
                           "_kd=False_augmix=False_T=4.0_alpha=0.7_st=full")
    adv_augmix_prefix   = ("resnet_pre=True_mode=finetune_freeze=False_ls=0.0"
                           "_kd=False_augmix=True_T=4.0_alpha=0.7_st=full")
    transfer_prefix = ("cnn_pre=False_mode=finetune_freeze=True_ls=0.0"
                       "_kd=False_augmix=False_T=4.0_alpha=0.7_st=full")

    # Clean accuracies
    clean_standard = 0.9070  # scratch ResNet
    clean_augmix   = 0.9547  # AugMix pretrained ResNet (used for adv evaluation)

    # Load corrupted results from scratch models
    standard_results = load_corrupted_results(standard_prefix)
    augmix_results   = load_corrupted_results(augmix_prefix)

    # Load adversarial results from pretrained fine-tuned models
    standard_adv = load_adversarial_results(adv_standard_prefix)
    augmix_adv   = load_adversarial_results(adv_augmix_prefix)

    transfer_results = load_transferability_results(transfer_prefix)

    # Generate plots
    plot_corrupted_comparison(standard_results, augmix_results,
                              clean_standard, clean_augmix)
    plot_adversarial_comparison(standard_adv, augmix_adv,
                                clean_standard, clean_augmix)
    plot_transferability(transfer_results)
    save_summary_csv(standard_results, augmix_results,
                     standard_adv, augmix_adv, transfer_results,
                     clean_standard, clean_augmix)

    print("\nAll HW2 summary plots saved to results/summary/")



if __name__ == "__main__":
    main()