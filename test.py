import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from parameters import Params
from train import get_transforms


def count_flops(model: nn.Module, params: Params) -> None:
    """
    Print the FLOPs and parameter count of a model using ptflops.

    Args:
        model: The PyTorch model to analyze.
        params: Configuration dataclass used to determine input resolution.
    """
    try:
        from ptflops import get_model_complexity_info
        input_res = (1, 28, 28) if params.dataset == "mnist" else (3, 32, 32)
        macs, n_params = get_model_complexity_info(
            model, input_res,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"\n=== Model Complexity ===")
        print(f"  MACs   : {macs}")
        print(f"  Params : {n_params}")
    except ImportError:
        print("ptflops not installed. Run: pip install ptflops")


@torch.no_grad()
def run_test(
    model:    nn.Module,
    params:   Params,
    device:   torch.device,
    run_name: str = "run",
) -> None:
    """
    Evaluate a trained model on the test set.

    Loads saved weights from params.save_path, counts FLOPs,
    runs inference, and prints overall and per-class accuracy.
    Saves results to a CSV file in the results/ directory.

    Args:
        model: The model to evaluate.
        params: Configuration dataclass.
        device: Device to run inference on.
        run_name: Label for saved test CSV.
    """
    import os
    import csv

    os.makedirs("results", exist_ok=True)

    tf = get_transforms(params, train=False)

    if params.dataset == "mnist":
        test_ds = datasets.MNIST(params.data_dir, train=False, download=True, transform=tf)
    else:
        test_ds = datasets.CIFAR10(params.data_dir, train=False, download=True, transform=tf)

    loader = DataLoader(test_ds, batch_size=params.batch_size,
                        shuffle=False, num_workers=params.num_workers)

    model.load_state_dict(torch.load(params.save_path, map_location=device))
    model.eval()

    # Count FLOPs before inference
    count_flops(model.cpu(), params)
    model.to(device)

    correct, n    = 0, 0
    class_correct = [0] * params.num_classes
    class_total   = [0] * params.num_classes

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n       += imgs.size(0)
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t]   += 1

    print(f"\n=== Test Results: {run_name} ===")
    print(f"Overall accuracy: {correct/n:.4f}  ({correct}/{n})\n")
    for i in range(params.num_classes):
        acc = class_correct[i] / class_total[i]
        print(f"  Class {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    csv_path = f"results/{run_name}_test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "correct", "total", "accuracy"])
        for i in range(params.num_classes):
            acc = class_correct[i] / class_total[i]
            writer.writerow([i, class_correct[i], class_total[i], f"{acc:.4f}"])
        writer.writerow(["overall", correct, n, f"{correct/n:.4f}"])
    print(f"Saved test CSV: {csv_path}")