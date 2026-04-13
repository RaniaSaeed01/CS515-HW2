import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

from models.CNN import SimpleCNN
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2


def count(model: nn.Module, name: str, input_res: tuple = (3, 32, 32)) -> None:
    """Print MACs and parameter count for a model."""
    macs, params = get_model_complexity_info(
        model, input_res,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"{name}")
    print(f"  MACs   : {macs}")
    print(f"  Params : {params}")
    print()


count(SimpleCNN(num_classes=10),                    "SimpleCNN")
count(ResNet(BasicBlock, [2,2,2,2], num_classes=10),"ResNet-18")
count(MobileNetV2(num_classes=10),                  "MobileNetV2")