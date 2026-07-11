import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class ProsTreeEngineTest(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("numpy is not installed")

    def test_entropy_selector_creates_partial_rollout_node(self):
        from fastgpro_based_pros.pros_tree import ProsRolloutRecord, ProsTreeConfig, ProsTreeEngine

        engine = ProsTreeEngine(
            original_data_len=2,
            config=ProsTreeConfig(
                selector="entropy",
                min_window_tokens=1,
                score_threshold=1.0,
                random_seed=7,
            ),
        )
        records = [
            ProsRolloutRecord(
                item=0,
                response_ids=list(range(20)),
                reward=1.0,
                response_mask=[1] * 20,
                partial_rollout_len=0,
                entropies=[0.0] * 8 + [9.0] + [0.0] * 11,
            )
        ]
        metrics = engine.update_data_source(records, step_num=1)
        self.assertEqual(len(engine), 3)
        child = engine.get_node(2)
        self.assertEqual(child.father_item, 0)
        self.assertEqual(child.partial_rollout_len, 8)
        self.assertEqual(metrics["dataset/partial_rollout_len_max"], 8.0)

    def test_mix_selector_uses_one_group_wide_value_percentile(self):
        from fastgpro_based_pros.pros_tree import ProsRolloutRecord, ProsTreeConfig, ProsTreeEngine

        engine = ProsTreeEngine(
            original_data_len=1,
            config=ProsTreeConfig(
                selector="mix",
                min_window_tokens=1,
                score_threshold=1.0,
                random_seed=7,
            ),
        )
        low_value_high_entropy = [0.0] * 14 + [100.0] + [0.0] * 5
        high_value_entropy = [0.0] * 14 + [10.0] + [0.0] * 5
        records = [
            ProsRolloutRecord(
                item=0,
                response_ids=list(range(20)),
                reward=1.0,
                response_mask=[1] * 20,
                partial_rollout_len=0,
                entropies=low_value_high_entropy,
                values=[float(value) for value in range(20)],
            ),
            ProsRolloutRecord(
                item=0,
                response_ids=list(range(100, 120)),
                reward=1.0,
                response_mask=[1] * 20,
                partial_rollout_len=0,
                entropies=high_value_entropy,
                values=[float(100 + value) for value in range(20)],
            ),
        ]

        engine.update_data_source(records, step_num=1)

        child = engine.get_node(1)
        self.assertEqual(child.partial_rollout_len, 14)
        self.assertEqual(child.partial_rollout[0], 100)

    def test_legacy_fallback_never_duplicates_an_original_ancestor(self):
        from fastgpro_based_pros.pros_tree import ProsTreeConfig, ProsTreeEngine

        engine = ProsTreeEngine(
            original_data_len=2,
            config=ProsTreeConfig(allow_fallback_fill=True, random_seed=7),
        )
        child = engine.create_new_node(
            father_node=engine.get_node(0),
            partial_rollout=[10, 11],
            step_num=1,
            score=1.0,
        )
        engine.psi[:] = 4.0
        engine.psi[child] = 0.0

        with self.assertRaisesRegex(ValueError, "Only 2 items collected for batch size 3"):
            engine.select_batch(batch_size=3, step_num=1)


if __name__ == "__main__":
    unittest.main()
