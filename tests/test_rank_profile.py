import unittest
from types import SimpleNamespace

import torch

from data_utils.path_utils import dataset_file_prefix
from nnm_module import compute_nnm_loss, layer_weight, select_mid_layers
from nnm_variants import compute_variant_loss
from rank_profile import rank_profile, rank_profile_loss_one_layer
from model_structure import resolve_transformer_layers


class RankProfileTest(unittest.TestCase):
    def test_resolve_qwen_layers_across_wrappers(self):
        layers = [object(), object()]
        raw = SimpleNamespace(model=SimpleNamespace(layers=layers))
        delegated = SimpleNamespace(
            base_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
        )
        peft = SimpleNamespace(
            base_model=SimpleNamespace(
                model=SimpleNamespace(model=SimpleNamespace(layers=layers))
            )
        )
        self.assertIs(resolve_transformer_layers(raw), layers)
        self.assertIs(resolve_transformer_layers(delegated), layers)
        self.assertIs(resolve_transformer_layers(peft), layers)

    def test_paper_layer_configuration(self):
        self.assertEqual(select_mid_layers(28, 4), [5, 11, 17, 23])
        self.assertEqual(layer_weight(11, 28), 1.0)

    def test_dataset_path_does_not_require_trailing_slash(self):
        self.assertEqual(
            dataset_file_prefix("/tmp/processed/model", "train", 0),
            "/tmp/processed/model/train_0",
        )
        self.assertEqual(
            dataset_file_prefix("/tmp/processed/model/", "train", 0),
            "/tmp/processed/model/train_0",
        )

    def test_translation_scale_and_orthogonal_invariance(self):
        torch.manual_seed(0)
        hidden = torch.randn(9, 5)
        q, _ = torch.linalg.qr(torch.randn(5, 5))
        transformed = 3.7 * hidden @ q + torch.randn(1, 5)
        torch.testing.assert_close(
            rank_profile(hidden), rank_profile(transformed), atol=2e-5, rtol=2e-5
        )

    def test_loss_is_projector_free_and_backpropagates(self):
        torch.manual_seed(1)
        student = torch.randn(7, 3, requires_grad=True)
        teacher = torch.randn(7, 6)
        loss = rank_profile_loss_one_layer(student, teacher)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_nuno_dispatch_matches_existing_entry_point(self):
        torch.manual_seed(2)
        projectors = torch.nn.ModuleList([torch.nn.Linear(3, 5, bias=False)])
        student_hidden = (torch.randn(2, 4, 3),)
        teacher_hidden = (torch.randn(2, 4, 5),)
        labels = torch.tensor([[1, 2, -100, -100], [3, 4, 5, -100]])
        kwargs = dict(
            projectors=projectors,
            s_hidden_states=student_hidden,
            t_hidden_states=teacher_hidden,
            labels=labels,
            student_layer_mapping=[0],
            teacher_layer_mapping=[0],
            t_centroids={0: torch.randn(4, 5)},
            R=torch.randn(5, 3),
            layer_weights={0: 0.3},
            ns_iters=3,
        )
        expected = compute_nnm_loss(**kwargs)
        actual = compute_variant_loss("nuno", **kwargs)
        torch.testing.assert_close(actual, expected)

    def test_rpt_shared_dispatch_accepts_no_nuno_state(self):
        student = torch.randn(1, 6, 3, requires_grad=True)
        teacher = torch.randn(1, 6, 8)
        loss = compute_variant_loss(
            "rpt",
            projectors=[None],
            s_hidden_states=(student,),
            t_hidden_states=(teacher,),
            labels=torch.tensor([[1, 2, 3, 4, -100, -100]]),
            student_layer_mapping=[0],
            teacher_layer_mapping=[0],
            t_centroids={},
            R=None,
            layer_weights={0: 1.0},
        )
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())


if __name__ == "__main__":
    unittest.main()
