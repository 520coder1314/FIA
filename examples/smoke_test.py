"""CPU-only interface check using generated tensors. Not an attack benchmark."""

import torch
from torch import nn
from fia.core import Proxy, fit_head, inject, screen


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    backbone = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    proxy = Proxy(backbone, 4, 2, "cifar10")
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 0, 1, 1])
    fit_head(proxy, images, labels, epochs=1, batch_size=2)
    with torch.no_grad():
        candidates = screen(proxy.features(images), labels, 0, 1)
    result = inject(
        proxy,
        images[:2],
        images[2:],
        0,
        1,
        steps=2,
        epsilon=0.05,
        quality_mode="stop_gradient",
    )
    assert candidates["undirected_total"] == 1
    assert result["patterns"].abs().max() <= 0.050001
    print("PASS: pair screening, frozen-weight head training with BN updates, and bounded injection.")
    print("Generated tensors only; no downstream VLM or attack-performance claim.")


if __name__ == "__main__":
    main()
