import unittest
from unittest.mock import patch

import torch

import cst_module
from cst_module import compute_cst_loss, compute_cst_transform
from nnm_variants import compute_variant_loss


class CSTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.gamma = torch.tensor([0.01, 0.1, 1.0, 10.0])

    def test_invariances(self):
        hidden = torch.randn(32, 12)
        q, _ = torch.linalg.qr(torch.randn(12, 12))
        reference = compute_cst_transform(hidden, self.gamma)
        torch.testing.assert_close(reference, compute_cst_transform(3.2 * hidden, self.gamma))
        torch.testing.assert_close(
            reference, compute_cst_transform(hidden + torch.randn(1, 12), self.gamma),
            atol=2e-5, rtol=2e-5,
        )
        torch.testing.assert_close(
            reference, compute_cst_transform(hidden @ q, self.gamma),
            atol=2e-5, rtol=2e-5,
        )

    def test_different_widths_gradient_and_teacher_stopgrad(self):
        student = torch.randn(1, 64, 128, requires_grad=True)
        teacher = torch.randn(1, 64, 512, requires_grad=True)
        labels = torch.ones(1, 64, dtype=torch.long)
        loss, diagnostics = compute_cst_loss(
            (student,), (teacher,), labels, [0], [0],
            max_tokens=64, fixed_gamma_grid=[0.1, 1.0],
        )
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(diagnostics["tokens"], 64)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad)

    def test_shared_dispatch_uses_no_projector_or_nuno_state(self):
        student = torch.randn(1, 10, 5, requires_grad=True)
        teacher = torch.randn(1, 10, 13)
        loss, diagnostics = compute_variant_loss(
            "cst", [None], (student,), (teacher,), torch.ones(1, 10, dtype=torch.long),
            [0], [0], {}, None, {},
            cst_options={"fixed_gamma_grid": [0.1, 1.0]},
        )
        loss.backward()
        self.assertEqual(diagnostics["valid_layers"], 1)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_sylvester_identity(self):
        x = torch.randn(7, 4)
        x = x - x.mean(0, keepdim=True)
        energy = x.square().sum()
        gamma = 2.3
        token = torch.eye(7) + gamma * (x @ x.mT) / energy
        feature = torch.eye(4) + gamma * (x.mT @ x) / energy
        torch.testing.assert_close(
            torch.linalg.slogdet(token).logabsdet,
            torch.linalg.slogdet(feature).logabsdet,
            atol=2e-6, rtol=2e-6,
        )

    def test_cholesky_matches_slogdet_and_batched_matches_loop(self):
        hidden = torch.randn(10, 6)
        batched = compute_cst_transform(hidden, self.gamma)
        looped = torch.stack([
            compute_cst_transform(hidden, gamma[None])[0] for gamma in self.gamma
        ])
        torch.testing.assert_close(batched, looped, atol=2e-6, rtol=2e-6)

        x = hidden - hidden.mean(0, keepdim=True)
        base = x.mT @ x / x.square().sum()
        matrix = torch.eye(6) + self.gamma[2] * base
        expected = torch.linalg.slogdet(matrix).logabsdet
        torch.testing.assert_close(batched[2], expected, atol=2e-6, rtol=2e-6)

    def test_degenerate_hidden_is_safely_skipped(self):
        student = torch.ones(1, 8, 4, requires_grad=True)
        teacher = torch.ones(1, 8, 9)
        loss, diagnostics = compute_cst_loss(
            (student,), (teacher,), torch.ones(1, 8, dtype=torch.long), [0], [0]
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(diagnostics["valid_layers"], 0)
        self.assertTrue(torch.isfinite(loss))

    def test_same_gamma_tensor_is_shared(self):
        student = torch.randn(1, 12, 5, requires_grad=True)
        teacher = torch.randn(1, 12, 9)
        seen = []
        original = cst_module.compute_cst_transform

        def recording_transform(hidden, gamma, **kwargs):
            seen.append(gamma)
            return original(hidden, gamma, **kwargs)

        with patch("cst_module.compute_cst_transform", side_effect=recording_transform):
            compute_cst_loss(
                (student,), (teacher,), torch.ones(1, 12, dtype=torch.long), [0], [0]
            )
        self.assertIs(seen[0], seen[1])

    def test_spectral_concentration_and_small_gamma_expansion(self):
        direction = torch.randn(32, 1)
        collapsed = direction @ torch.ones(1, 8)
        isotropic, _ = torch.linalg.qr(torch.randn(32, 8))
        phi_collapsed = compute_cst_transform(collapsed, torch.tensor([1.0]))
        phi_isotropic = compute_cst_transform(isotropic, torch.tensor([1.0]))
        self.assertGreater((phi_collapsed - phi_isotropic).abs().item(), 1e-3)

        hidden = torch.randn(12, 7)
        x = hidden - hidden.mean(0, keepdim=True)
        base = x.mT @ x / x.square().sum()
        gamma = 1e-3
        approximation = gamma - 0.5 * gamma**2 * torch.trace(base @ base)
        actual = compute_cst_transform(hidden, torch.tensor([gamma]))[0]
        torch.testing.assert_close(actual, approximation, atol=2e-6, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
