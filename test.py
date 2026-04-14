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


def evaluate_corrupted(
    model:    nn.Module,
    params:   Params,
    device:   torch.device,
    run_name: str = "run",
) -> None:
    """
    Evaluate model on CIFAR-10-C corrupted test set.

    Loads a specific corruption type and severity level from the
    CIFAR-10-C dataset, runs inference, and reports accuracy.

    Args:
        model: Trained model to evaluate.
        params: Configuration dataclass containing corruption settings.
        device: Device to run on.
        run_name: Label for saved results CSV.
    """
    import csv
    import numpy as np
    import os

    data_path   = os.path.join(params.cifar10c_dir,
                               f"{params.corruption_type}.npy")
    labels_path = os.path.join(params.cifar10c_dir, "labels.npy")

    if not os.path.exists(data_path):
        print(f"CIFAR-10-C not found at {params.cifar10c_dir}. "
              f"Download from https://zenodo.org/record/2535967")
        return

    data   = np.load(data_path)
    labels = np.load(labels_path)

    # Each file has 5 severity levels stacked: 10000 images each
    start  = (params.corruption_severity - 1) * 10000
    end    = start + 10000
    data   = data[start:end]
    labels = labels[start:end]

    # Normalize to match training preprocessing
    mean = np.array(params.mean)
    std  = np.array(params.std)
    data   = data.astype(np.float32) / 255.0
    mean_np = np.array(params.mean).reshape(1, 1, 1, 3)
    std_np  = np.array(params.std).reshape(1, 1, 1, 3)
    data   = (data - mean_np) / std_np
    data   = torch.tensor(data).float().permute(0, 3, 1, 2)
    labels = torch.tensor(labels).long()

    dataset = torch.utils.data.TensorDataset(data, labels)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=params.batch_size, shuffle=False)

    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(1)
            correct += preds.eq(lbls).sum().item()
            n       += imgs.size(0)

    acc = correct / n
    print(f"\n=== Corrupted Test ({params.corruption_type} "
          f"severity={params.corruption_severity}) ===")
    print(f"Accuracy: {acc:.4f}  ({correct}/{n})")

    os.makedirs("results", exist_ok=True)
    csv_path = (f"results/{run_name}_corrupted_"
                f"{params.corruption_type}_s{params.corruption_severity}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["corruption", "severity", "accuracy"])
        writer.writerow([params.corruption_type,
                         params.corruption_severity, f"{acc:.4f}"])
    print(f"Saved: {csv_path}")


def evaluate_adversarial(
    model:    nn.Module,
    params:   Params,
    device:   torch.device,
    run_name: str = "run",
) -> None:
    """
    Evaluate model robustness against PGD adversarial attacks.

    Runs PGD-20 with both L-inf (eps=4/255) and L2 (eps=0.25) norms
    and reports accuracy on adversarial examples.

    Args:
        model: Trained model to evaluate.
        params: Configuration dataclass containing attack settings.
        device: Device to run on.
        run_name: Label for saved results CSV.
    """
    import csv
    import os
    from attacks import pgd_attack_linf, pgd_attack_l2
    from torchvision import datasets

    os.makedirs("results", exist_ok=True)

    tf      = get_transforms(params, train=False)
    test_ds = datasets.CIFAR10(params.data_dir, train=False,
                               download=True, transform=tf)
    loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=params.num_workers)

    model.eval()
    results = {}

    for attack_type in ["pgd_linf", "pgd_l2"]:
        correct, n = 0, 0
        print(f"\nRunning {attack_type}...")

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if attack_type == "pgd_linf":
                adv = pgd_attack_linf(
                    model, imgs, labels,
                    params.pgd_eps_linf,
                    params.pgd_alpha_linf,
                    params.pgd_steps, device)
            else:
                adv = pgd_attack_l2(
                    model, imgs, labels,
                    params.pgd_eps_l2,
                    params.pgd_alpha_l2,
                    params.pgd_steps, device)

            with torch.no_grad():
                preds = model(adv).argmax(1)
            correct += preds.eq(labels).sum().item()
            n       += imgs.size(0)

        acc = correct / n
        results[attack_type] = acc
        print(f"  {attack_type} accuracy: {acc:.4f}")

    csv_path = f"results/{run_name}_adversarial.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "accuracy"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.4f}"])
    print(f"Saved: {csv_path}")


def evaluate_transferability(
    teacher:  nn.Module,
    student:  nn.Module,
    params:   Params,
    device:   torch.device,
    run_name: str = "run",
) -> None:
    """
    Test adversarial transferability from teacher to student model.

    Generates PGD-20 L-inf adversarial examples using the teacher model,
    then evaluates both teacher and student accuracy on those examples.
    This tests whether adversarial examples crafted for one model
    transfer to fool a different model.

    Args:
        teacher: The source model used to generate adversarial examples.
        student: The target model to test transferability on.
        params: Configuration dataclass containing attack settings.
        device: Device to run on.
        run_name: Label for saved results CSV.
    """
    import csv
    import os
    from attacks import pgd_attack_linf
    from torchvision import datasets

    os.makedirs("results", exist_ok=True)

    tf      = get_transforms(params, train=False)
    test_ds = datasets.CIFAR10(params.data_dir, train=False,
                               download=True, transform=tf)
    loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=params.num_workers)

    teacher.eval()
    student.eval()

    teacher_correct, student_correct, n = 0, 0, 0

    print("\nRunning adversarial transferability test...")
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Generate adversarial examples using teacher
        adv = pgd_attack_linf(
            teacher, imgs, labels,
            params.pgd_eps_linf,
            params.pgd_alpha_linf,
            params.pgd_steps, device)

        # Test on both teacher and student
        with torch.no_grad():
            teacher_preds = teacher(adv).argmax(1)
            student_preds = student(adv).argmax(1)

        teacher_correct += teacher_preds.eq(labels).sum().item()
        student_correct += student_preds.eq(labels).sum().item()
        n += imgs.size(0)

    teacher_acc = teacher_correct / n
    student_acc = student_correct / n

    print(f"\n=== Adversarial Transferability Results ===")
    print(f"  Teacher accuracy on teacher adv examples: {teacher_acc:.4f}")
    print(f"  Student accuracy on teacher adv examples: {student_acc:.4f}")

    csv_path = f"results/{run_name}_transferability.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy"])
        writer.writerow(["teacher_on_teacher_adv", f"{teacher_acc:.4f}"])
        writer.writerow(["student_on_teacher_adv", f"{student_acc:.4f}"])
    print(f"Saved: {csv_path}")


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