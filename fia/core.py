"""Paper-aligned primitives. New runs only; not a reconstruction of old results."""

import torch
from torch import nn
from torch.nn import functional as F


class Proxy(nn.Module):
    """Frozen weights; BN updates during head fitting, eval buffers during injection."""

    def __init__(self, backbone, dim, classes, preprocess):
        super().__init__()
        if preprocess not in ("cifar10", "imagenet"):
            raise ValueError("Choose checkpoint-compatible preprocessing explicitly")
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = nn.Linear(dim, classes)
        self.preprocess = preprocess
        mean, std = (
            ([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
            if preprocess == "cifar10"
            else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def feature_input(self, x):
        if self.preprocess == "imagenet":
            if x.shape[-2:] != (224, 224):
                x = F.interpolate(x, (256, 256), mode="bilinear", align_corners=False)
                x = x[:, :, 16:240, 16:240]
        elif x.shape[-2:] != (32, 32):
            raise ValueError(
                "cifar10 preprocessing requires 32x32 inputs; do not silently resize"
            )
        return (x - self.mean) / self.std

    def features(self, x):
        return self.backbone(self.feature_input(x))

    def classifier_input(self, x):
        x = F.interpolate(x, (256, 256), mode="bilinear", align_corners=False)
        x = x[:, :, 16:240, 16:240]
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (x - mean) / std

    def forward(self, x):
        return self.head(self.backbone(self.classifier_input(x)))


def fit_head(proxy, x, y, epochs=150, batch_size=32):
    proxy.train()
    opt = torch.optim.SGD(
        proxy.head.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    # Recompute minibatches: fitting updates shared backbone BN buffers.
    for _ in range(epochs):
        for ix in torch.randperm(len(y), device=y.device).split(batch_size):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(proxy(x[ix]), y[ix]).backward()
            opt.step()
    proxy.eval()


def screen(features, labels, low=0.30, high=0.90):
    if not 0 <= low < high <= 1:
        raise ValueError("Require 0 <= low < high <= 1")
    classes = labels.unique(sorted=True)
    if len(classes) < 2:
        raise ValueError("At least two classes required")
    centers = torch.stack([features[labels == c].mean(0) for c in classes]).double()
    d = torch.cdist(centers, centers)
    ij = torch.triu_indices(len(classes), len(classes), 1)
    values = d[ij[0], ij[1]]
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite distances")
    lo, hi = torch.quantile(
        values, torch.tensor([low, high], device=d.device, dtype=d.dtype)
    )
    pairs = []
    for i, j in ij.T.tolist():
        v = float(d[i, j])
        pairs.append(
            {
                "source": int(classes[i]),
                "target": int(classes[j]),
                "distance": v,
                "eligible": bool(lo <= d[i, j] <= hi),
            }
        )
    return {
        "quantiles": [low, high],
        "cutoffs": [float(lo), float(hi)],
        "undirected_total": len(pairs),
        "undirected_kept": sum(p["eligible"] for p in pairs),
        "pairs": pairs,
    }


def channel_gate_mask(proxy, x, label, floor=0.05):
    """Executed paper channel-gating mask; explicitly NOT standard Grad-CAM++."""
    if not 0 <= floor <= 1:
        raise ValueError("Invalid mask floor")
    layer = next(
        m for m in reversed(list(proxy.backbone.modules())) if isinstance(m, nn.Conv2d)
    )
    captured = []
    hook = layer.register_forward_hook(lambda m, i, o: captured.append(o))
    try:
        raw = x.detach().clone().requires_grad_(True)
        score = proxy(proxy.feature_input(raw))[:, label].sum()
        act = captured[0]
        (grad,) = torch.autograd.grad(score, act)
        gate = (grad.clamp_min(0).sum((2, 3), keepdim=True) > 0).to(act.dtype)
        cam = (act * gate).sum(1, keepdim=True)
        cam = cam - cam.amin((2, 3), keepdim=True)
        cam = cam / (cam.amax((2, 3), keepdim=True) + 1e-8)
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam - cam.amin((2, 3), keepdim=True)
        cam = cam / (cam.amax((2, 3), keepdim=True) + 1e-8)
        return (floor + (1 - floor) * cam).detach()
    finally:
        hook.remove()


def appearance(a, b):
    """PSNR/SSIM hinges: 7x7 uniform SSIM, sample covariance, RGB mean.

    Matches skimage's default SSIM window/covariance convention on RGB images.
    The caller detaches the score for the manuscript optimization profile.
    """
    if min(a.shape[-2:]) < 7:
        raise ValueError("SSIM requires at least 7 pixels per dimension")
    mse = (a - b).square().mean((1, 2, 3)).clamp_min(1e-12)
    psnr = -10 * torch.log10(mse)
    filt = lambda x: F.avg_pool2d(x, kernel_size=7, stride=1)
    ma, mb = filt(a), filt(b)
    va, vb = (filt(a * a) - ma * ma) * (49 / 48), (filt(b * b) - mb * mb) * (49 / 48)
    cov = (filt(a * b) - ma * mb) * (49 / 48)
    ss = (
        ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2))
        / ((ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2))
    ).mean((1, 2, 3))
    return (0.5 * F.relu(30 - psnr) + 0.5 * F.relu(0.9 - ss)).mean()


def objective(proxy, clean, mask, delta, target, center, quality_mode):
    poisoned = (clean + mask * delta).clamp(0, 1)
    z = proxy.features(poisoned)
    cls = F.cross_entropy(
        proxy(poisoned), torch.full((len(z),), target, device=z.device, dtype=torch.long)
    )
    direction = (
        (z / (z.norm(dim=1, keepdim=True) + 1e-12)
         - center / (center.norm() + 1e-12))
        .square()
        .sum(1)
        .mean()
    )
    norm = (z.norm(dim=1) / (center.norm() + 1e-12) - 1).square().mean()
    raw = ((z - center).square().sum(1) + 1e-12).sqrt().mean()
    align = 0.6 * direction + 0.4 * norm + 0.01 * raw
    quality = appearance(poisoned, clean)
    if quality_mode == "stop_gradient":
        quality = quality.detach()
    elif quality_mode != "differentiable":
        raise ValueError("Choose quality mode explicitly")
    tv = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean() + (
        delta[:, :, :, 1:] - delta[:, :, :, :-1]
    ).abs().mean()
    return 0.45 * cls + 0.45 * align + 0.1 * quality + 5e-4 * tv


def inject(
    proxy,
    clean,
    target_images,
    source,
    target,
    steps=4500,
    epsilon=0.15,
    quality_mode="stop_gradient",
):
    if steps < 1 or epsilon <= 0:
        raise ValueError("Positive steps and epsilon required")
    proxy.eval()
    mask = channel_gate_mask(proxy, clean, source)
    with torch.no_grad():
        center = proxy.features(target_images).mean(0)
    delta = nn.Parameter(torch.zeros_like(clean[:1]))
    opt = torch.optim.Adam([delta], lr=0.05, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.95, patience=1000, min_lr=1e-5
    )
    best = float("inf")
    best_delta = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = objective(proxy, clean, mask, delta, target, center, quality_mode)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite objective")
        score = float(loss.detach())
        if score < best:
            best = score
            best_delta = delta.detach().clone()
        # No accumulation of unused head gradients.
        (delta.grad,) = torch.autograd.grad(loss, delta)
        nn.utils.clip_grad_norm_([delta], 50)
        opt.step()
        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)
        scheduler.step(score)
    # Include the final projected update in best-iterate selection.
    with torch.no_grad():
        score = float(
            objective(proxy, clean, mask, delta, target, center, quality_mode)
        )
        if score < best:
            best_delta = delta.detach().clone()
            best = score
    patterns = mask * best_delta
    return {
        "images": (clean + patterns).clamp(0, 1).detach(),
        "patterns": patterns.detach(),
        "delta": best_delta,
        "masks": mask,
        "best_score": best,
    }


def apply_bank(images, patterns, base_indices):
    """Cyclic assignment by base-class index; caller preserves original evaluation labels."""
    if len(patterns) == 0:
        raise ValueError("Empty trigger bank")
    if images.shape[1:] != patterns.shape[1:]:
        raise ValueError("Pattern dimensions differ")
    return (images + patterns[base_indices % len(patterns)]).clamp(0, 1)
