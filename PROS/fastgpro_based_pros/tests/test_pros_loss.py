import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class ProsLossTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

    def test_grpo_advantages_center_by_group(self):
        import torch

        from fastgpro_based_pros.pros_loss import compute_grpo_advantages

        rewards = torch.tensor([1.0, 3.0, 2.0, 2.0])
        mask = torch.ones((4, 2))
        advantages = compute_grpo_advantages(rewards, mask, [0, 0, 1, 1], normalize_by_std=False)
        expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        self.assertTrue(torch.allclose(advantages, expected))

    def test_gpg_alpha_matches_reference_behavior(self):
        import torch

        from fastgpro_based_pros.pros_loss import compute_gpg_advantages

        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
        mask = torch.ones((4, 1))
        advantages = compute_gpg_advantages(rewards, mask, [0, 0, 1, 1])
        # alpha = batch / nonzero = 4 / 2 = 2, group means are 0.5.
        expected = torch.tensor([[-1.0], [1.0], [-1.0], [1.0]])
        self.assertTrue(torch.allclose(advantages, expected))


if __name__ == "__main__":
    unittest.main()
