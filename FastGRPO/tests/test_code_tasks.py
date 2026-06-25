import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper.multitask import load_multitask_QAs, render_messages
from helper.rewards import compute_reward_from_example


class CodeTaskTest(unittest.TestCase):
    def test_code_placeholder_reward_scores_static_signals(self):
        completion = """```python
def solve():
    return 1
```"""
        example = {
            "reward_type": "code",
            "language": "python",
            "entry_point": "solve",
            "expected_substrings": ["return 1"],
        }
        self.assertEqual(compute_reward_from_example(completion, example), 1.0)

    def test_records_config_loads_code_task(self):
        config = {
            "samples_per_epoch": 1,
            "tasks": [
                {
                    "id": "code_inline",
                    "prompt_type": "code",
                    "language": "python",
                    "records": [
                        {
                            "prompt": "Write solve() that returns 1.",
                            "entry_point": "solve",
                            "tests": "assert solve() == 1",
                        }
                    ],
                }
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as f:
            json.dump(config, f)
            f.flush()
            samples = load_multitask_QAs(f.name, seed=123)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["prompt_type"], "code")
        self.assertEqual(sample["reward_type"], "code")
        self.assertEqual(sample["language"], "python")
        self.assertEqual(sample["entry_point"], "solve")
        self.assertIn("tests", sample)
        messages = render_messages(sample)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("coding problem", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
