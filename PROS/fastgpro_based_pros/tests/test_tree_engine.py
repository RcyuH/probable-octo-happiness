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


if __name__ == "__main__":
    unittest.main()
