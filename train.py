import copy
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import Params


def get_transforms(params: Params, train: bool = True) -> transforms.Compose:
    """
    Build a torchvision transform pipeline for the given dataset and split.

    For CIFAR-10 training, applies random cropping and horizontal flipping.
    When using pretrained models in finetune mode, resizes images to 224x224
    to match ImageNet input dimensions. When augmix is enabled, applies
    AugMix augmentation to improve robustness against distribution shifts.

    Args:
        params: Configuration dataclass containing dataset, mean, std,
                pretrained, transfer_mode, and augmix settings.
        train: If True, apply training augmentations.

    Returns:
        A composed torchvision transform pipeline.
    """
    mean, std = params.mean, params.std
    resize = (params.pretrained and params.transfer_mode == "finetune"
              and params.dataset == "cifar10")

    if params.dataset == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        if train:
            t = [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip()]
            if params.augmix:
                t.append(transforms.AugMix(
                    severity=params.augmix_severity,
                    mixture_width=params.augmix_width,
                    chain_depth=params.augmix_depth,
                    alpha=params.augmix_alpha,
                ))
            if resize:
                t.append(transforms.Resize(224))
            t += [transforms.ToTensor(), transforms.Normalize(mean, std)]
            return transforms.Compose(t)
        else:
            t = []
            if resize:
                t.append(transforms.Resize(224))
            t += [transforms.ToTensor(), transforms.Normalize(mean, std)]
            return transforms.Compose(t)


def get_loaders(params: Params) -> Tuple[DataLoader, DataLoader]:
    """
    Build and return training and validation DataLoaders.

    Args:
        params: Configuration dataclass containing dataset and data settings.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    train_tf = get_transforms(params, train=True)
    val_tf   = get_transforms(params, train=False)

    if params.dataset == "mnist":
        train_ds = datasets.MNIST(params.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(params.data_dir, train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(params.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(params.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=params.batch_size,
                              shuffle=True,  num_workers=params.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=params.batch_size,
                              shuffle=False, num_workers=params.num_workers)
    return train_loader, val_loader


def get_criterion(params: Params) -> nn.Module:
    """
    Return the loss function based on params.

    Uses CrossEntropyLoss with optional label smoothing. Label smoothing
    prevents overconfidence by distributing a small probability mass
    uniformly across non-target classes.

    Args:
        params: Configuration dataclass containing label_smoothing.

    Returns:
        A loss function module.
    """
    return nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)


def distillation_loss(
    student_logits:   torch.Tensor,
    teacher_logits:   torch.Tensor,
    labels:           torch.Tensor,
    temperature:      float,
    alpha:            float,
    criterion:        nn.Module,
    soft_target_mode: str = "full",
) -> torch.Tensor:
    """
    Compute the knowledge distillation loss.

    Combines a soft target loss (student learns from teacher probability
    distribution) with a hard target loss (student learns from ground truth).

    For soft_target_mode='full': standard KD using the full teacher softmax
    distribution as soft targets.

    For soft_target_mode='true_only': assign teacher's confidence for the
    true class to that class only; distribute remaining probability equally
    across all other classes. This encodes per-example difficulty from
    the teacher without transferring inter-class relationships.

    Args:
        student_logits: Raw logits from the student model (B, C).
        teacher_logits: Raw logits from the teacher model (B, C).
        labels: Ground truth class indices (B,).
        temperature: Softens probability distributions. Higher = softer.
        alpha: Weight for the soft distillation loss (1-alpha for hard loss).
        criterion: Hard target loss function (e.g. CrossEntropyLoss).
        soft_target_mode: 'full' or 'true_only'.

    Returns:
        Combined scalar loss value.
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    if soft_target_mode == "true_only":
        batch_size, num_classes = teacher_probs.size()
        true_probs   = teacher_probs[range(batch_size), labels].unsqueeze(1)
        remaining    = (1.0 - true_probs) / (num_classes - 1)
        soft_targets = remaining.expand_as(teacher_probs).clone()
        soft_targets[range(batch_size), labels] = true_probs.squeeze(1)
    else:
        soft_targets = teacher_probs

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(student_log_probs, soft_targets,
                         reduction="batchmean") * (temperature ** 2)
    hard_loss = criterion(student_logits, labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    params:    Params,
    teacher:   Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """
    Train the model for a single epoch.

    If params.distillation is True and a teacher is provided, uses
    distillation_loss instead of the standard criterion.

    Args:
        model: The student model to train.
        loader: DataLoader providing training batches.
        optimizer: Optimizer for weight updates.
        criterion: Hard target loss function.
        device: Device to run on.
        params: Configuration dataclass.
        teacher: Optional frozen teacher model for distillation.

    Returns:
        A tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)

        if params.distillation and teacher is not None:
            with torch.no_grad():
                teacher_out = teacher(imgs)
            loss = distillation_loss(
                out, teacher_out, labels,
                params.temperature, params.alpha,
                criterion, params.soft_target_mode,
            )
        else:
            loss = criterion(out, labels)

        if params.l1_lambda > 0.0:
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss = loss + params.l1_lambda * l1_penalty

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % params.log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]  "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on a validation DataLoader.

    Args:
        model: The model to evaluate.
        loader: DataLoader providing validation batches.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        A tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

    return total_loss / n, correct / n


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    params:    Params,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Construct a learning rate scheduler based on params.scheduler.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        params: Configuration dataclass.

    Returns:
        An LRScheduler instance, or None if params.scheduler is 'none'.
    """
    if params.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif params.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs)
    return None


def run_training(
    model:    nn.Module,
    params:   Params,
    device:   torch.device,
    run_name: str = "run",
    teacher:  Optional[nn.Module] = None,
) -> None:
    """
    Execute the full training loop with validation, checkpointing,
    early stopping, CSV logging, and loss/accuracy plotting.

    Args:
        model: The student (or standalone) model to train.
        params: Configuration dataclass with all training hyperparameters.
        device: Device to run on.
        run_name: Label used for naming saved CSV and plot files.
        teacher: Optional frozen teacher model for knowledge distillation.
    """
    import os
    import csv
    import matplotlib.pyplot as plt

    os.makedirs("results", exist_ok=True)

    train_loader, val_loader = get_loaders(params)
    criterion = get_criterion(params)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    scheduler = build_scheduler(optimizer, params)

    best_acc         = 0.0
    best_weights     = None
    patience_counter = 0

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    for epoch in range(1, params.epochs + 1):
        print(f"\nEpoch {epoch}/{params.epochs}")

        tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer,
                                            criterion, device, params, teacher)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc         = val_acc
            best_weights     = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_weights, params.save_path)
            print(f"  Saved best model (val_acc={best_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{params.early_stop_patience}")
            if patience_counter >= params.early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")

    # Save CSV
    csv_path = f"results/{run_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history.keys())
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({k: history[k][i] for k in history})
    print(f"Saved CSV: {csv_path}")

    # Save plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(run_name)
    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss")
    ax1.plot(history["epoch"], history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss");   ax1.legend()
    ax2.plot(history["epoch"], history["train_acc"],  label="Train Acc")
    ax2.plot(history["epoch"], history["val_acc"],    label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend()
    plot_path = f"results/{run_name}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot: {plot_path}")