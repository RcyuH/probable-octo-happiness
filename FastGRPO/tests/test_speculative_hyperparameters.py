import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SpeculativeHyperparametersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from helper.speculative_hyperparameters import get_adaptive_hyperparameters

        cls.get_adaptive_hyperparameters = staticmethod(get_adaptive_hyperparameters)

    def test_raises_clear_error_when_verification_capacity_is_too_small(self):
        with self.assertRaisesRegex(ValueError, "verification_capacity is too small"):
            self.get_adaptive_hyperparameters(
                bsz=161,
                verification_capacity=160,
                max_draft_token_length=5,
                max_draft_k=8,
                max_verification_num=160,
                min_draft_token_length=3,
                draft_token_length_c=0.75,
            )

    def test_returns_minimal_valid_budget(self):
        self.assertEqual(
            self.get_adaptive_hyperparameters(
                bsz=80,
                verification_capacity=160,
                max_draft_token_length=5,
                max_draft_k=8,
                max_verification_num=160,
                min_draft_token_length=3,
                draft_token_length_c=0.75,
            ),
            (3, 1, 1),
        )

    def test_preserves_default_budget_behavior(self):
        self.assertEqual(
            self.get_adaptive_hyperparameters(
                bsz=32,
                verification_capacity=160,
                max_draft_token_length=5,
                max_draft_k=8,
                max_verification_num=160,
                min_draft_token_length=3,
                draft_token_length_c=0.75,
            ),
            (3, 4, 4),
        )


if __name__ == "__main__":
    unittest.main()
