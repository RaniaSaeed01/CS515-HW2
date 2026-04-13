import torch
import torch.nn as nn


def pgd_attack_linf(
    model:  nn.Module,
    imgs:   torch.Tensor,
    labels: torch.Tensor,
    eps:    float,
    alpha:  float,
    steps:  int,
    device: torch.device,
) -> torch.Tensor:
    """
    PGD attack with L-infinity norm constraint (Madry et al. 2018).

    Generates adversarial examples by iteratively perturbing input images
    in the direction of the gradient of the loss, clipping perturbations
    to stay within an L-inf ball of radius eps around the original image.

    Args:
        model: The target model to attack.
        imgs: Clean input images (B, C, H, W) in [0, 1].
        labels: Ground truth labels (B,).
        eps: Maximum L-inf perturbation radius.
        alpha: Step size per iteration.
        steps: Number of PGD iterations.
        device: Device to run on.

    Returns:
        Adversarial images of the same shape as imgs.
    """
    criterion = nn.CrossEntropyLoss()
    adv_imgs  = imgs.clone().detach()

    # Random initialization within epsilon ball
    adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-eps, eps)
    adv_imgs = torch.clamp(adv_imgs, 0, 1).detach()

    for _ in range(steps):
        adv_imgs.requires_grad_(True)
        outputs = model(adv_imgs)
        loss    = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv_imgs = adv_imgs + alpha * adv_imgs.grad.sign()
            delta    = torch.clamp(adv_imgs - imgs, -eps, eps)
            adv_imgs = torch.clamp(imgs + delta, 0, 1).detach()

    return adv_imgs


def pgd_attack_l2(
    model:  nn.Module,
    imgs:   torch.Tensor,
    labels: torch.Tensor,
    eps:    float,
    alpha:  float,
    steps:  int,
    device: torch.device,
) -> torch.Tensor:
    """
    PGD attack with L2 norm constraint.

    Similar to L-inf PGD but projects perturbations onto an L2 ball
    of radius eps after each step.

    Args:
        model: The target model to attack.
        imgs: Clean input images (B, C, H, W) in [0, 1].
        labels: Ground truth labels (B,).
        eps: Maximum L2 perturbation radius.
        alpha: Step size per iteration.
        steps: Number of PGD iterations.
        device: Device to run on.

    Returns:
        Adversarial images of the same shape as imgs.
    """
    criterion  = nn.CrossEntropyLoss()
    adv_imgs   = imgs.clone().detach()
    batch_size = imgs.size(0)

    # Random initialization within L2 ball
    delta  = torch.empty_like(adv_imgs).normal_()
    d_flat = delta.view(batch_size, -1)
    n      = d_flat.norm(p=2, dim=1).view(batch_size, 1, 1, 1)
    r      = torch.zeros_like(n).uniform_(0, 1)
    delta  = eps * delta * r / (n + 1e-8)
    adv_imgs = torch.clamp(adv_imgs + delta, 0, 1).detach()

    for _ in range(steps):
        adv_imgs.requires_grad_(True)
        outputs = model(adv_imgs)
        loss    = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad        = adv_imgs.grad
            grad_flat   = grad.view(batch_size, -1)
            grad_norm   = grad_flat.norm(p=2, dim=1).view(batch_size, 1, 1, 1)
            grad_normed = grad / (grad_norm + 1e-8)

            adv_imgs   = adv_imgs + alpha * grad_normed
            delta      = adv_imgs - imgs
            delta_flat = delta.view(batch_size, -1)
            delta_norm = delta_flat.norm(p=2, dim=1).view(batch_size, 1, 1, 1)
            delta      = eps * delta / torch.clamp(delta_norm, min=eps)
            adv_imgs   = torch.clamp(imgs + delta, 0, 1).detach()

    return adv_imgs