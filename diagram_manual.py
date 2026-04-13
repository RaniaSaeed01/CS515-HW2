import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


def draw_mlp_diagram(
    hidden_sizes: list,
    input_size:   int  = 784,
    num_classes:  int  = 10,
    activation:   str  = "ReLU",
    use_bn:       bool = True,
    dropout:      float = 0.3,
) -> None:
    """
    Draw a clean MLP architecture diagram and save to results/.

    Args:
        hidden_sizes: List of hidden layer widths.
        input_size: Number of input features.
        num_classes: Number of output classes.
        activation: Activation function name for labels.
        use_bn: Whether BatchNorm is used.
        dropout: Dropout probability for labels.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#f9f9f9")

    # Define layer info
    layers = (
        [("Input\n784", "#AED6F1")]
        + [(f"Hidden\n{h}\nLinear→{'BN→' if use_bn else ''}{activation}→Drop({dropout})", "#A9DFBF")
           for h in hidden_sizes]
        + [(f"Output\n{num_classes}", "#F9E79F")]
    )

    n      = len(layers)
    xs     = [i * (9 / (n - 1)) + 0.5 for i in range(n)]
    y      = 3.0
    width  = 1.2
    height = 1.8

    for i, ((label, color), x) in enumerate(zip(layers, xs)):
        # Draw box
        box = mpatches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width, height,
            boxstyle="round,pad=0.1",
            linewidth=1.5,
            edgecolor="#2C3E50",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#2C3E50")

        # Draw arrow to next layer
        if i < n - 1:
            ax.annotate(
                "", xy=(xs[i + 1] - width / 2, y),
                xytext=(x + width / 2, y),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5),
            )

    # Title and legend
    ax.set_title("MLP Architecture — MNIST Classification",
                 fontsize=13, fontweight="bold", pad=15, color="#2C3E50")

    legend_elements = [
        mpatches.Patch(facecolor="#AED6F1", edgecolor="#2C3E50", label="Input layer"),
        mpatches.Patch(facecolor="#A9DFBF", edgecolor="#2C3E50", label="Hidden layer"),
        mpatches.Patch(facecolor="#F9E79F", edgecolor="#2C3E50", label="Output layer"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              ncol=3, fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig("results/mlp_architecture.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/mlp_architecture.png")


if __name__ == "__main__":
    draw_mlp_diagram(
        hidden_sizes=[512, 256, 128],
        input_size=784,
        num_classes=10,
        activation="ReLU",
        use_bn=True,
        dropout=0.3,
    )