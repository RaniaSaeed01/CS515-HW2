from dataclasses import dataclass, field
from typing import List, Tuple
import argparse


@dataclass
class Params:
    """
    Configuration dataclass for HW1a (MLP/MNIST) and HW1b
    (Transfer Learning and Knowledge Distillation on CIFAR-10).

    Attributes:
        dataset: Dataset to use ('mnist' or 'cifar10').
        data_dir: Directory to download/load data from.
        num_workers: Number of parallel DataLoader workers.
        mean: Normalization mean per channel.
        std: Normalization std per channel.
        input_size: Flattened input dimensionality.
        num_classes: Number of output classes.
        model: Model architecture to use.
        hidden_sizes: MLP hidden layer widths.
        dropout: Dropout probability.
        activation: Activation function ('relu' or 'gelu').
        use_bn: Whether to use BatchNorm1d in MLP.
        vgg_depth: VGG variant depth string.
        resnet_layers: Number of blocks per ResNet stage.
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_backbone: Whether to freeze backbone and train head only.
        transfer_mode: 'finetune' (resize images) or 'scratch' (modify conv).
        label_smoothing: Label smoothing coefficient (0.0 = off).
        distillation: Whether to use knowledge distillation.
        teacher_path: Path to saved teacher model weights.
        temperature: Distillation temperature for softening distributions.
        alpha: Weight of soft loss in distillation (1-alpha for hard loss).
        soft_target_mode: 'full' for standard KD, 'true_only' for HW1b part 3.
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per batch.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization coefficient.
        l1_lambda: L1 regularization coefficient.
        scheduler: LR scheduler type ('step', 'cosine', or 'none').
        early_stop_patience: Epochs to wait before early stopping.
        seed: Random seed for reproducibility.
        device: Device string ('cpu', 'cuda', or 'mps').
        save_path: File path to save the best model weights.
        log_interval: Batch interval for printing training progress.
        mode: Pipeline mode ('train', 'test', or 'both').
        augmix: Whether to use AugMix augmentation during training.
        augmix_severity: Severity of AugMix augmentations (1-10).
        augmix_width: Number of augmentation chains in AugMix.
        augmix_depth: Depth of each chain (-1 for random).
        augmix_alpha: Mixing coefficient for AugMix.
        pgd_eps_linf: L-infinity epsilon for PGD attack.
        pgd_eps_l2: L2 epsilon for PGD attack.
        pgd_steps: Number of PGD iterations.
        pgd_alpha_linf: Step size for L-inf PGD.
        pgd_alpha_l2: Step size for L2 PGD.
        corrupted: Whether to evaluate on CIFAR-10-C.
        corruption_type: Type of corruption to evaluate on.
        corruption_severity: Severity level of corruption (1-5).
        cifar10c_dir: Path to CIFAR-10-C dataset directory.
    """

    # Data
    dataset:     str               = "cifar10"
    data_dir:    str               = "./data"
    num_workers: int               = 2
    mean:        Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    std:         Tuple[float, ...] = (0.2023, 0.1994, 0.2010)
    input_size:  int               = 3072
    num_classes: int               = 10

    # Model
    model:         str       = "resnet"
    hidden_sizes:  List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout:       float     = 0.3
    activation:    str       = "relu"
    use_bn:        bool      = True
    vgg_depth:     str       = "16"
    resnet_layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])

    # Transfer learning
    pretrained:      bool = True
    freeze_backbone: bool = True
    transfer_mode:   str  = "finetune"

    # Label smoothing
    label_smoothing: float = 0.0

    # Knowledge distillation
    distillation:     bool  = False
    teacher_path:     str   = "best_teacher.pth"
    temperature:      float = 4.0
    alpha:            float = 0.7
    soft_target_mode: str   = "full"

    # AugMix
    augmix:              bool  = False
    augmix_severity:     int   = 3
    augmix_width:        int   = 3
    augmix_depth:        int   = -1
    augmix_alpha:        float = 1.0

    # Adversarial attacks
    pgd_eps_linf:        float = 4/255
    pgd_eps_l2:          float = 0.25
    pgd_steps:           int   = 20
    pgd_alpha_linf:      float = 1/255
    pgd_alpha_l2:        float = 0.05

    # CIFAR-10-C
    corrupted:           bool  = False
    corruption_type:     str   = "fog"
    corruption_severity: int   = 1
    cifar10c_dir:        str   = "./data/CIFAR-10-C"

    # Training
    epochs:              int   = 20
    batch_size:          int   = 128
    learning_rate:       float = 1e-3
    weight_decay:        float = 1e-4
    l1_lambda:           float = 0.0
    scheduler:           str   = "step"
    early_stop_patience: int   = 5

    # Misc
    seed:         int = 42
    device:       str = "cpu"
    save_path:    str = "best_model.pth"
    log_interval: int = 100
    mode:         str = "both"


def get_params() -> Params:
    """
    Parse command-line arguments and return a populated Params dataclass.

    Returns:
        Params: A dataclass instance with all configuration values set.

    Example:
        $ python main.py --model resnet --dataset cifar10 --pretrained --epochs 20
    """
    parser = argparse.ArgumentParser(description="HW1b: Transfer Learning and Knowledge Distillation")

    # Data / model
    parser.add_argument("--mode", choices=[
        "train", "test", "both",
        "test_corrupted",
        "test_adversarial",
        "visualize_adv",
        "transferability",
    ], default="both")

    parser.add_argument("--dataset", choices=["mnist", "cifar10"],                          default="cifar10")
    parser.add_argument("--model",   choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"], default="resnet")
    parser.add_argument("--vgg_depth",     choices=["11", "13", "16", "19"],                default="16")
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"))

    # Transfer learning
    parser.add_argument("--transfer_mode",   choices=["finetune", "scratch"], default="finetune")
    parser.add_argument("--pretrained",      action="store_true", default=True)
    parser.add_argument("--no_pretrained",   action="store_true", default=False)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze",       action="store_true", default=False)

    # Label smoothing
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # Knowledge distillation
    parser.add_argument("--distillation",     action="store_true", default=False)
    parser.add_argument("--teacher_path",     type=str,   default="best_teacher.pth")
    parser.add_argument("--temperature",      type=float, default=4.0)
    parser.add_argument("--alpha",            type=float, default=0.7)
    parser.add_argument("--soft_target_mode", choices=["full", "true_only"], default="full")

    # AugMix
    parser.add_argument("--augmix",             action="store_true", default=False)
    parser.add_argument("--augmix_severity",    type=int,   default=3)
    parser.add_argument("--augmix_width",       type=int,   default=3)
    parser.add_argument("--augmix_depth",       type=int,   default=-1)
    parser.add_argument("--augmix_alpha",       type=float, default=1.0)

    # Adversarial attacks
    parser.add_argument("--pgd_eps_linf",       type=float, default=4/255)
    parser.add_argument("--pgd_eps_l2",         type=float, default=0.25)
    parser.add_argument("--pgd_steps",          type=int,   default=20)
    parser.add_argument("--pgd_alpha_linf",     type=float, default=1/255)
    parser.add_argument("--pgd_alpha_l2",       type=float, default=0.05)

    # CIFAR-10-C
    parser.add_argument("--corrupted",          action="store_true", default=False)
    parser.add_argument("--corruption_type",    type=str,   default="fog")
    parser.add_argument("--corruption_severity",type=int,   default=1)
    parser.add_argument("--cifar10c_dir",       type=str,   default="./data/CIFAR-10-C")

    # Training
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--l1_lambda",   type=float, default=0.0)
    parser.add_argument("--scheduler",   choices=["step", "cosine", "none"], default="step")
    parser.add_argument("--early_stop_patience", type=int, default=5)

    # Misc
    parser.add_argument("--device",    type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default="best_model.pth")

    args = parser.parse_args()

    if args.dataset == "mnist":
        input_size = 784
        mean = (0.1307,)
        std  = (0.3081,)
    else:
        input_size = 3072
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)

    return Params(
        dataset=args.dataset,
        mean=mean,
        std=std,
        input_size=input_size,
        model=args.model,
        vgg_depth=args.vgg_depth,
        resnet_layers=args.resnet_layers,
        transfer_mode=args.transfer_mode,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.no_freeze,
        label_smoothing=args.label_smoothing,
        distillation=args.distillation,
        teacher_path=args.teacher_path,
        temperature=args.temperature,
        alpha=args.alpha,
        soft_target_mode=args.soft_target_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda,
        scheduler=args.scheduler,
        early_stop_patience=args.early_stop_patience,
        device=args.device,
        save_path=args.save_path,
        mode=args.mode,
        augmix=args.augmix,
        augmix_severity=args.augmix_severity,
        augmix_width=args.augmix_width,
        augmix_depth=args.augmix_depth,
        augmix_alpha=args.augmix_alpha,
        pgd_eps_linf=args.pgd_eps_linf,
        pgd_eps_l2=args.pgd_eps_l2,
        pgd_steps=args.pgd_steps,
        pgd_alpha_linf=args.pgd_alpha_linf,
        pgd_alpha_l2=args.pgd_alpha_l2,
        corrupted=args.corrupted,
        corruption_type=args.corruption_type,
        corruption_severity=args.corruption_severity,
        cifar10c_dir=args.cifar10c_dir,
    )