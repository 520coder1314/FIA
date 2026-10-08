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
    objective,
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

    def test_backbone_weights_frozen_bn_updates_only_during_fit(self):
        weights = {k: v.detach().clone() for k, v in self.proxy.backbone.named_parameters()}
        before = self.proxy.backbone.bn.running_mean.clone()
        fit_head(self.proxy, self.x, self.y, epochs=2, batch_size=2)
        for k, v in self.proxy.backbone.named_parameters():
            self.assertTrue(torch.equal(weights[k], v), k)
        self.assertFalse(torch.equal(before, self.proxy.backbone.bn.running_mean))
        self.assertEqual(int(self.proxy.backbone.bn.num_batches_tracked), 4)
        buffers = {k: v.clone() for k, v in self.proxy.backbone.named_buffers()}
        inject(self.proxy, self.x[:2], self.x[2:], 0, 1, steps=2)
        for k, v in self.proxy.backbone.named_buffers():
            self.assertTrue(torch.equal(buffers[k], v), k)

    def test_separate_preprocessing_paths(self):
        self.proxy.eval()
        seen = []
        hook = self.proxy.backbone.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().clone()))
        self.proxy(self.x)
        self.proxy.features(self.x)
        channel_gate_mask(self.proxy, self.x, 0)
        hook.remove()
        self.assertEqual(seen[0].shape[-2:], (224, 224))
        self.assertEqual(seen[1].shape[-2:], (32, 32))
        expected = self.proxy.classifier_input(self.proxy.feature_input(self.x))
        self.assertTrue(torch.equal(seen[2], expected))

    def test_objective_matches_equations_and_detaches_quality(self):
        self.proxy.eval()
        clean = self.x[:2]
        mask = torch.ones(2, 1, 32, 32)
        delta = torch.full_like(clean[:1], 0.04, requires_grad=True)
        with torch.no_grad():
            center = self.proxy.features(self.x[2:]).mean(0)
        poisoned = (clean + mask * delta).clamp(0, 1)
        z = self.proxy.features(poisoned)
        cls = torch.nn.functional.cross_entropy(self.proxy(poisoned), torch.ones(2, dtype=torch.long))
        direction = ((z / (z.norm(dim=1, keepdim=True) + 1e-12) - center / (center.norm() + 1e-12)) ** 2).sum(1).mean()
        norm = (z.norm(dim=1) / (center.norm() + 1e-12) - 1).square().mean()
        raw = ((z - center).square().sum(1) + 1e-12).sqrt().mean()
        expected = .45 * cls + .45 * (.6 * direction + .4 * norm + .01 * raw)
        actual = objective(self.proxy, clean, mask, delta, 1, center, "stop_gradient")
        self.assertTrue(torch.allclose(actual, expected + .1 * appearance(poisoned, clean).detach()))
        ga = torch.autograd.grad(actual, delta, retain_graph=True)[0]
        ge = torch.autograd.grad(expected, delta)[0]
        self.assertTrue(torch.allclose(ga, ge))

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
