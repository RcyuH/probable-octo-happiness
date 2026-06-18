import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastgpro_based_pros.pros_config import ProsConfig


class ProsConfigTest(unittest.TestCase):
    def test_multitask_cli_fields_are_coerced(self):
        cfg = ProsConfig.from_args(
            [
                "--dry-run",
                "true",
                "--task-config",
                "FastGRPO/configs/multitask_rlvr.example.json",
                "--task-samples-per-epoch",
                "17",
                "--generation-backend",
                "target",
                "--train-draft",
                "false",
            ]
        )
        self.assertTrue(cfg.dry_run)
        self.assertEqual(cfg.task_config, "FastGRPO/configs/multitask_rlvr.example.json")
        self.assertEqual(cfg.task_samples_per_epoch, 17)
        self.assertEqual(cfg.generation_backend, "target")
        self.assertFalse(cfg.train_draft)


if __name__ == "__main__":
    unittest.main()
