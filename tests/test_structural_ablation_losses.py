import unittest

import torch

from nnm_variants import compute_variant_loss


class StructuralAblationLossTest(unittest.TestCase):
    def test_projector_free_losses_support_width_mismatch_and_gradient(self):
        for variant in ("gram", "cka", "normalized_spectrum", "direct_spectrum"):
            student = torch.randn(1, 12, 5, requires_grad=True)
            teacher = torch.randn(1, 12, 11, requires_grad=True)
            loss = compute_variant_loss(
                variant, [None], (student,), (teacher,),
                torch.ones(1, 12, dtype=torch.long), [0], [0], {}, None, {},
                cst_options={"max_tokens": 8},
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss), variant)
            self.assertTrue(torch.isfinite(student.grad).all(), variant)
            self.assertIsNone(teacher.grad, variant)

    def test_hidden_mse_uses_learned_projector(self):
        student = torch.randn(1, 9, 4, requires_grad=True)
        teacher = torch.randn(1, 9, 7, requires_grad=True)
        projector = torch.nn.Linear(4, 7, bias=False)
        loss = compute_variant_loss(
            "hidden_mse", [projector], (student,), (teacher,),
            torch.ones(1, 9, dtype=torch.long), [0], [0], {}, None, {},
        )
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertTrue(torch.isfinite(projector.weight.grad).all())
        self.assertIsNone(teacher.grad)

    def test_identical_geometry_has_near_zero_intrinsic_losses(self):
        hidden = torch.randn(1, 10, 6)
        for variant in ("gram", "cka", "normalized_spectrum", "direct_spectrum"):
            loss = compute_variant_loss(
                variant, [None], (hidden,), (hidden.clone(),),
                torch.ones(1, 10, dtype=torch.long), [0], [0], {}, None, {},
            )
            self.assertLess(abs(loss.item()), 2e-6, variant)


if __name__ == "__main__":
    unittest.main()
