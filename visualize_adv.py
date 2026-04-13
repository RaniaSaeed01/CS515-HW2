import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets
from torch.utils.data import DataLoader

from parameters import Params
from train import get_transforms
from attacks import pgd_attack_linf
from gradcam import GradCAM, plot_gradcam


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def run_gradcam(
    model:    nn.Module,
    params:   Params,
    device:   torch.device,
    save_dir: str = "results/gradcam",
) -> None:
    """
    Generate Grad-CAM visualizations for misclassified adversarial samples.

    Finds samples where the clean image is correctly classified but the
    adversarial version is misclassified, then plots Grad-CAM overlays
    for both the clean and adversarial images side by side.

    Args:
        model: Trained ResNet model.
        params: Configuration dataclass.
        device: Device to run on.
        save_dir: Directory to save Grad-CAM plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    tf      = get_transforms(params, train=False)
    test_ds = datasets.CIFAR10(params.data_dir, train=False,
                               download=True, transform=tf)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=True)

    # Hook into last conv layer of ResNet
    target_layer = model.layer4[-1].conv2
    gradcam      = GradCAM(model, target_layer)

    found = 0
    for imgs, labels in loader:
        if found >= 2:
            break

        imgs, labels = imgs.to(device), labels.to(device)

        # Check clean prediction
        model.eval()
        with torch.no_grad():
            pred_clean = model(imgs).argmax(1).item()

        if pred_clean != labels.item():
            continue  # skip already misclassified clean images

        # Generate adversarial example
        adv = pgd_attack_linf(
            model, imgs, labels,
            params.pgd_eps_linf,
            params.pgd_alpha_linf,
            params.pgd_steps, device)

        with torch.no_grad():
            pred_adv = model(adv).argmax(1).item()

        if pred_adv == labels.item():
            continue  # skip if attack did not fool the model

        # Generate Grad-CAM heatmaps
        cam_clean = gradcam.generate(imgs)
        cam_adv   = gradcam.generate(adv)

        plot_gradcam(
            imgs, adv, cam_clean, cam_adv,
            labels.item(), pred_clean, pred_adv,
            CIFAR10_CLASSES,
            save_path=f"{save_dir}/gradcam_sample_{found + 1}.png",
        )
        found += 1

    print(f"Saved {found} Grad-CAM plots to {save_dir}/")


def run_tsne(
    model:     nn.Module,
    params:    Params,
    device:    torch.device,
    save_path: str = "results/tsne_adversarial.png",
    n_samples: int = 500,
) -> None:
    """
    Visualize clean and adversarial samples using t-SNE.

    Extracts penultimate layer features for both clean and adversarial
    images then plots their t-SNE embeddings colored by class label,
    showing how adversarial perturbations shift feature representations.

    Args:
        model: Trained ResNet model.
        params: Configuration dataclass.
        device: Device to run on.
        save_path: Path to save the t-SNE plot.
        n_samples: Number of samples to use for t-SNE.
    """
    tf      = get_transforms(params, train=False)
    test_ds = datasets.CIFAR10(params.data_dir, train=False,
                               download=True, transform=tf)
    loader  = DataLoader(test_ds, batch_size=64,
                         shuffle=False, num_workers=params.num_workers)

    model.eval()
    features_clean = []
    features_adv   = []
    all_labels     = []
    count          = 0

    def extract_features(x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate layer features from ResNet."""
        x = model.conv1(x)
        x = model.bn1(x)
        x = torch.relu(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return x.view(x.size(0), -1)

    for imgs, labels in loader:
        if count >= n_samples:
            break

        imgs, labels = imgs.to(device), labels.to(device)

        adv = pgd_attack_linf(
            model, imgs, labels,
            params.pgd_eps_linf,
            params.pgd_alpha_linf,
            params.pgd_steps, device)

        with torch.no_grad():
            feat_clean = extract_features(imgs).cpu().numpy()
            feat_adv   = extract_features(adv).cpu().numpy()

        features_clean.append(feat_clean)
        features_adv.append(feat_adv)
        all_labels.append(labels.cpu().numpy())
        count += len(imgs)

    features_clean = np.concatenate(features_clean)[:n_samples]
    features_adv   = np.concatenate(features_adv)[:n_samples]
    all_labels     = np.concatenate(all_labels)[:n_samples]

    print("Running t-SNE (this may take a few minutes)...")
    combined = np.concatenate([features_clean, features_adv])
    tsne     = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(combined)

    n         = len(features_clean)
    emb_clean = embedded[:n]
    emb_adv   = embedded[n:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("t-SNE: Clean vs Adversarial Feature Representations", fontsize=13)

    sc1 = axes[0].scatter(emb_clean[:, 0], emb_clean[:, 1],
                          c=all_labels, cmap="tab10", s=10, alpha=0.7)
    plt.colorbar(sc1, ax=axes[0], ticks=range(10))
    axes[0].set_title("Clean Samples")
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")

    sc2 = axes[1].scatter(emb_adv[:, 0], emb_adv[:, 1],
                          c=all_labels, cmap="tab10", s=10, alpha=0.7)
    plt.colorbar(sc2, ax=axes[1], ticks=range(10))
    axes[1].set_title("Adversarial Samples (PGD-20 L-inf)")
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot: {save_path}")