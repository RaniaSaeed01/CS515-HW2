import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron for MNIST classification.

    Builds a configurable deep network using ModuleList for hidden layers
    and Sequential for input flattening and output projection. Each hidden
    layer optionally includes BatchNorm1d (before activation), a configurable
    activation function, and Dropout.

    Args:
        input_size: Number of input features (784 for MNIST).
        hidden_sizes: List of hidden layer widths.
        num_classes: Number of output classes.
        dropout: Dropout probability applied after each activation.
        activation: Activation function to use ('relu' or 'gelu').
        use_bn: Whether to insert BatchNorm1d before each activation.

    Example:
        >>> model = MLP(input_size=784, hidden_sizes=[512, 256], num_classes=10)
        >>> out = model(torch.randn(32, 1, 28, 28))
        >>> out.shape
        torch.Size([32, 10])
    """

    def __init__(
        self,
        input_size:   int       = 784,
        hidden_sizes: List[int] = [512, 256, 128],
        num_classes:  int       = 10,
        dropout:      float     = 0.3,
        activation:   str       = "relu",
        use_bn:       bool      = True,
    ) -> None:
        super().__init__()

        self.flatten = nn.Sequential(nn.Flatten())

        self.hidden_layers = nn.ModuleList()
        in_dim = input_size

        for h in hidden_sizes:
            block: List[nn.Module] = [nn.Linear(in_dim, h)]
            if use_bn:
                block.append(nn.BatchNorm1d(h))
            block.append(nn.ReLU() if activation == "relu" else nn.GELU())
            block.append(nn.Dropout(dropout))
            self.hidden_layers.append(nn.Sequential(*block))
            in_dim = h

        self.output = nn.Sequential(nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (B, 1, 28, 28) or (B, 784).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output(x)