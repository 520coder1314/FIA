import unittest
import torch
from torch import nn
from fia.core import (
    Proxy,
    fit_head,
    screen,
    channel_gate_mask,
    appearance,
    inject,
    apply_bank,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.conv(x)).relu().mean((2, 3))


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.proxy = Proxy(Tiny(), 4, 2, "cifar10")
        self.x = torch.rand(4, 3, 32, 32)
        self.y = torch.tensor([0, 0, 1, 1])

    def test_backbone_buffers_and_weights_frozen(self):
        before = {k: v.clone() for k, v in self.proxy.backbone.state_dict().items()}
        head = self.proxy.head.weight.detach().clone()
        fit_head(self.proxy, self.x, self.y, epochs=2, batch_size=2)
        for k, v in before.items():
            self.assertTrue(torch.equal(v, self.proxy.backbone.state_dict()[k]), k)
        self.assertFalse(torch.equal(head, self.proxy.head.weight))

    def test_gradient_quality_and_mask(self):
        a = (self.x + 0.1).clamp(0, 1).requires_grad_()
        (g,) = torch.autograd.grad(appearance(a, self.x), a)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(float(g.abs().sum()), 0)
        m = channel_gate_mask(self.proxy, self.x[:2], 0)
        self.assertEqual(m.shape, (2, 1, 32, 32))
        self.assertGreaterEqual(float(m.min()), 0.049999)
        self.assertLessEqual(float(m.max()), 1)

    def test_screen_counts_unique_undirected(self):
        r = screen(torch.tensor([[0.0], [1.0], [4.0]]), torch.tensor([0, 1, 2]), 0, 1)
        self.assertEqual(r["undirected_total"], 3)
        self.assertEqual(r["undirected_kept"], 3)
        self.assertEqual([p["distance"] for p in r["pairs"]], [1, 4, 3])

    def test_injection_bounded_and_reuse(self):
        for mode in ["stop_gradient", "differentiable"]:
            r = inject(
                self.proxy,
                self.x[:2],
                self.x[2:],
                0,
                1,
                steps=2,
                epsilon=0.05,
                quality_mode=mode,
            )
            self.assertLessEqual(float(r["patterns"].abs().max()), 0.050001)
            self.assertEqual(r["patterns"].shape, self.x[:2].shape)
            result = apply_bank(self.x, r["patterns"], torch.arange(4))
            self.assertTrue(
                torch.allclose(result[0], (self.x[0] + r["patterns"][0]).clamp(0, 1))
            )
            self.assertTrue(torch.isfinite(result).all())

    def test_nonfinite_distances_rejected(self):
        with self.assertRaises(ValueError):
            screen(torch.tensor([[float("nan")], [1.0]]), torch.tensor([0, 1]))


if __name__ == "__main__":
    unittest.main()
