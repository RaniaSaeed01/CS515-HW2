from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Visualizes which regions of an input image a CNN focuses on
    when making a prediction, by computing gradients of the target
    class score with respect to the last convolutional feature maps.

    Args:
        model: The CNN model to visualize.
        target_layer: The convolutional layer to hook into.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        """Save forward pass activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        """Save backward pass gradients."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        img:       torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given image.

        Args:
            img: Input image tensor of shape (1, C, H, W).
            class_idx: Target class index. If None, uses predicted class.

        Returns:
            Heatmap as a numpy array of shape (H, W) in [0, 1].
        """
        self.model.eval()
        output = self.model(img)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def plot_gradcam(
    img_clean:   torch.Tensor,
    img_adv:     torch.Tensor,
    cam_clean:   np.ndarray,
    cam_adv:     np.ndarray,
    label:       int,
    pred_clean:  int,
    pred_adv:    int,
    class_names: List[str],
    save_path:   str,
    mean:        tuple = (0.4914, 0.4822, 0.4465),
    std:         tuple = (0.2023, 0.1994, 0.2010),
) -> None:
    """
    Plot side-by-side Grad-CAM overlays for clean and adversarial images.

    Args:
        img_clean: Clean image tensor (1, C, H, W).
        img_adv: Adversarial image tensor (1, C, H, W).
        cam_clean: Grad-CAM heatmap for clean image.
        cam_adv: Grad-CAM heatmap for adversarial image.
        label: True class index.
        pred_clean: Predicted class for clean image.
        pred_adv: Predicted class for adversarial image.
        class_names: List of class name strings.
        save_path: Path to save the figure.
        mean: Normalization mean used during training.
        std: Normalization std used during training.
    """
    import cv2

    def denorm(t: torch.Tensor) -> np.ndarray:
        t = t.squeeze().permute(1, 2, 0).cpu().numpy()
        t = t * np.array(std) + np.array(mean)
        return np.clip(t, 0, 1)

    def overlay(img_np: np.ndarray, cam: np.ndarray) -> np.ndarray:
        h, w      = img_np.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap   = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        return np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)

    img_clean_np = denorm(img_clean)
    img_adv_np   = denorm(img_adv)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"True: {class_names[label]}  |  "
        f"Clean pred: {class_names[pred_clean]}  |  "
        f"Adv pred: {class_names[pred_adv]}"
    )

    axes[0].imshow(img_clean_np)
    axes[0].set_title("Clean Image")
    axes[0].axis("off")

    axes[1].imshow(overlay(img_clean_np, cam_clean))
    axes[1].set_title("Grad-CAM (Clean)")
    axes[1].axis("off")

    axes[2].imshow(img_adv_np)
    axes[2].set_title("Adversarial Image")
    axes[2].axis("off")

    axes[3].imshow(overlay(img_adv_np, cam_adv))
    axes[3].set_title("Grad-CAM (Adversarial)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM: {save_path}")