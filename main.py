import random
import ssl
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from parameters import Params, get_params
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from train import run_training
from test  import run_test
from test import evaluate_corrupted, evaluate_adversarial

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """
    Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_model(params: Params) -> nn.Module:
    """
    Build and return the appropriate model based on params.

    For transfer learning with pretrained=True:
      - 'finetune' mode: loads ImageNet weights, optionally freezes backbone,
        replaces classifier head. Images are resized to 224x224 in transforms.
      - 'scratch' mode: loads ImageNet weights but modifies the first conv
        layer to accept 32x32 input directly, fine-tunes the full network.

    For pretrained=False: builds the model architecture from scratch
    with random initialization.

    Args:
        params: Configuration dataclass.

    Returns:
        Configured nn.Module ready for training.
    """
    nc = params.num_classes

    if params.model == "mlp":
        return MLP(
            input_size   = params.input_size,
            hidden_sizes = params.hidden_sizes,
            num_classes  = nc,
            dropout      = params.dropout,
            activation   = params.activation,
            use_bn       = params.use_bn,
        )

    if params.model == "cnn":
        if params.dataset == "mnist":
            return MNIST_CNN(norm=nn.BatchNorm2d, num_classes=nc)
        else:
            return SimpleCNN(num_classes=nc)

    if params.model == "resnet":
        if params.pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            if params.transfer_mode == "scratch":
                # Modify first conv for 32x32 CIFAR-10 input
                model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
            if params.freeze_backbone and params.transfer_mode == "finetune":
                for p in model.parameters():
                    p.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, nc)
            return model
        else:
            return ResNet(BasicBlock, params.resnet_layers, num_classes=nc)

    if params.model == "vgg":
        if params.pretrained:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            if params.freeze_backbone and params.transfer_mode == "finetune":
                for p in model.features.parameters():
                    p.requires_grad = False
            model.classifier[6] = nn.Linear(4096, nc)
            return model
        else:
            return VGG(dept=params.vgg_depth, num_class=nc)

    if params.model == "mobilenet":
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {params.model}")


def load_teacher(params: Params, device: torch.device) -> nn.Module:
    """
    Load a saved teacher model for knowledge distillation.

    Builds a torchvision ResNet18, loads saved weights, freezes all
    parameters, and sets eval mode.

    Args:
        params: Configuration dataclass containing teacher_path.
        device: Device to load onto.

    Returns:
        Frozen teacher model in eval mode.
    """
    from torchvision import models
    teacher = models.resnet18(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, params.num_classes)
    teacher.load_state_dict(torch.load(params.teacher_path, map_location=device))
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher.to(device)


def main() -> None:
    """
    Entry point for the HW2 pipeline.

    Handles transfer learning, knowledge distillation, corrupted
    evaluation, adversarial evaluation, and Grad-CAM visualization.
    Auto-generates a descriptive run name from key hyperparameters.
    """
    params = get_params()
    set_seed(params.seed)

    device = torch.device(
        params.device if torch.cuda.is_available() else
        "mps"         if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")
    print(f"Dataset: {params.dataset}  |  Model: {params.model}")
    print(f"Pretrained: {params.pretrained}  |  Transfer mode: {params.transfer_mode}")
    print(f"Label smoothing: {params.label_smoothing}  |  Distillation: {params.distillation}")
    print(f"AugMix: {params.augmix}  |  Mode: {params.mode}")

    run_name = (
        f"{params.model}"
        f"_pre={params.pretrained}"
        f"_mode={params.transfer_mode}"
        f"_freeze={params.freeze_backbone}"
        f"_ls={params.label_smoothing}"
        f"_kd={params.distillation}"
        f"_augmix={params.augmix}"
        f"_T={params.temperature}"
        f"_alpha={params.alpha}"
        f"_st={params.soft_target_mode}"
    )

    model            = build_model(params).to(device)
    teacher: Optional[nn.Module] = None

    if params.distillation:
        print(f"Loading teacher from: {params.teacher_path}")
        teacher = load_teacher(params, device)

    if params.mode in ("train", "both"):
        run_training(model, params, device, run_name=run_name, teacher=teacher)

    if params.mode in ("test", "both"):
        run_test(model, params, device, run_name=run_name)

    if params.mode == "test_corrupted":
        model.load_state_dict(torch.load(params.save_path, map_location=device))
        model.eval()
        evaluate_corrupted(model, params, device, run_name=run_name)

    if params.mode == "test_adversarial":
        model.load_state_dict(torch.load(params.save_path, map_location=device))
        model.eval()
        evaluate_adversarial(model, params, device, run_name=run_name)

    if params.mode == "transferability":
        from torchvision import models
        # Load teacher
        teacher_model = models.resnet18(weights=None)
        teacher_model.fc = nn.Linear(teacher_model.fc.in_features, params.num_classes)
        teacher_model.load_state_dict(torch.load(params.teacher_path, map_location=device))
        teacher_model = teacher_model.to(device)
        teacher_model.eval()

        # Load student
        model.load_state_dict(torch.load(params.save_path, map_location=device))
        model.eval()

        from test import evaluate_transferability
        evaluate_transferability(teacher_model, model, params, device, run_name=run_name)



    if params.mode == "visualize_adv":
        model.load_state_dict(torch.load(params.save_path, map_location=device))
        model.eval()
        from visualize_adv import run_gradcam, run_tsne
        run_gradcam(model, params, device)
        run_tsne(model, params, device)


if __name__ == "__main__":
    main()