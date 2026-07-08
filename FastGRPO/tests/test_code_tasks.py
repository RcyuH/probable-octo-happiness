import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper.multitask import (
    TaskWeightedBatchSampler,
    compute_multitask_reward_debug,
    has_explicit_task_weights,
    load_multitask_QAs,
    render_messages,
)
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

    def test_code_unit_test_reward_executes_assert_tests(self):
        completion = """```python
def solve():
    return 1
```"""
        example = {
            "reward_type": "code_unit_test",
            "language": "python",
            "entry_point": "solve",
            "tests": "assert solve() == 1",
            "timeout_seconds": 1.0,
        }
        self.assertEqual(compute_reward_from_example(completion, example), 1.0)

    def test_code_unit_test_reward_rejects_wrong_solution(self):
        completion = """```python
def solve():
    return 2
```"""
        example = {
            "reward_type": "code_unit_test",
            "language": "python",
            "entry_point": "solve",
            "tests": "assert solve() == 1",
            "timeout_seconds": 1.0,
        }
        self.assertEqual(compute_reward_from_example(completion, example), 0.0)

    def test_code_unit_test_debug_reports_assertion_failures(self):
        completion = """```python
def solve():
    return 2
```"""
        example = {
            "reward_type": "code_unit_test",
            "language": "python",
            "entry_point": "solve",
            "tests": "assert solve() == 1",
            "timeout_seconds": 1.0,
        }
        detail = compute_multitask_reward_debug(completion, example)
        self.assertEqual(detail["reward"], 0.0)
        self.assertEqual(detail["error_type"], "assertion")
        self.assertFalse(detail["passed"])

    def test_code_unit_test_debug_reports_empty_tests(self):
        completion = """```python
def solve():
    return 1
```"""
        example = {
            "reward_type": "code_unit_test",
            "language": "python",
            "entry_point": "solve",
            "tests": "",
            "timeout_seconds": 1.0,
        }
        detail = compute_multitask_reward_debug(completion, example)
        self.assertEqual(detail["reward"], 0.0)
        self.assertEqual(detail["error_type"], "empty_tests")

    def test_code_unit_test_reward_executes_stdin_stdout_tests(self):
        completion = """```python
import sys
n = int(sys.stdin.read())
print(n * 2)
```"""
        example = {
            "reward_type": "code_unit_test",
            "language": "python",
            "test_type": "stdin_stdout",
            "tests": [{"input": "21", "output": "42"}],
            "timeout_seconds": 1.0,
        }
        self.assertEqual(compute_reward_from_example(completion, example), 1.0)

    def test_code_unit_test_reward_runs_unittest_classes(self):
        completion = """```python
def solve():
    return 1
```"""
        example = {
            "reward_type": "code_unit_test",
            "language": "python",
            "entry_point": "solve",
            "tests": """
import unittest

class SolveTest(unittest.TestCase):
    def test_solve(self):
        self.assertEqual(solve(), 1)
""",
            "timeout_seconds": 1.0,
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

    def test_records_config_loads_code_unit_test_task_fields(self):
        config = {
            "samples_per_epoch": 1,
            "tasks": [
                {
                    "id": "code_inline",
                    "prompt_type": "code",
                    "reward_type": "code_unit_test",
                    "language": "python",
                    "test_type": "unit",
                    "timeout_seconds": 1.5,
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

        self.assertEqual(samples[0]["reward_type"], "code_unit_test")
        self.assertEqual(samples[0]["test_type"], "unit")
        self.assertEqual(samples[0]["timeout_seconds"], 1.5)

    def test_task_weighted_batch_sampler_uses_weights_per_batch(self):
        config = {
            "samples_per_epoch": 16,
            "tasks": [
                {
                    "id": "math",
                    "weight": 5,
                    "prompt_type": "math",
                    "records": [
                        {"question": f"math {idx}", "answer": str(idx)}
                        for idx in range(16)
                    ],
                },
                {
                    "id": "code",
                    "weight": 3,
                    "prompt_type": "code",
                    "records": [
                        {
                            "prompt": f"code {idx}",
                            "entry_point": "solve",
                            "tests": "assert solve() == 1",
                        }
                        for idx in range(16)
                    ],
                },
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as f:
            json.dump(config, f)
            f.flush()
            samples = load_multitask_QAs(f.name, seed=123)

        self.assertTrue(has_explicit_task_weights(samples))
        sampler = TaskWeightedBatchSampler(samples, batch_size=8, seed=123)
        batches = list(sampler)

        self.assertEqual(len(batches), 2)
        for batch in batches:
            task_ids = [samples[idx]["task_id"] for idx in batch]
            self.assertEqual(task_ids.count("math"), 5)
            self.assertEqual(task_ids.count("code"), 3)


if __name__ == "__main__":
    unittest.main()
