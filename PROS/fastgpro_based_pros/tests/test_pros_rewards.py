import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "FastGRPO"))

from helper.multitask import compute_multitask_reward_debug


class ProsSharedRewardRegressionTest(unittest.TestCase):
    def test_basic_reward_dispatchers(self):
        cases = (
            (
                "exact_match",
                "  The   Answer  ",
                {"answer": "the answer"},
                1.0,
            ),
            (
                "contains",
                "The computed result is forty two.",
                {"answer": "FORTY TWO"},
                1.0,
            ),
            (
                "regex",
                "answer=42",
                {"pattern": r"answer=\d+"},
                1.0,
            ),
            (
                "format_only",
                "reasoning\n</think>\nfinal answer",
                {},
                1.0,
            ),
            (
                "zero",
                "anything",
                {},
                0.0,
            ),
        )

        for reward_type, completion, extra, expected in cases:
            with self.subTest(reward_type=reward_type):
                detail = compute_multitask_reward_debug(
                    completion,
                    {"reward_type": reward_type, **extra},
                )
                self.assertEqual(detail["reward"], expected)
                self.assertEqual(detail["reward_type"], reward_type)
                self.assertEqual(detail["passed"], expected > 0)

        zero_detail = compute_multitask_reward_debug("anything", {"reward_type": "zero"})
        self.assertEqual(zero_detail["error_type"], "zero_reward")

    def test_custom_reward_callable_supports_colon_and_dotted_specs(self):
        module_name = "_pros_test_custom_reward_module"
        module = types.ModuleType(module_name)
        calls = []

        def score(*, completion, example):
            calls.append((completion, example["scale"]))
            return example["scale"] if completion == "pass" else 0.0

        module.score = score
        with mock.patch.dict(sys.modules, {module_name: module}):
            colon_detail = compute_multitask_reward_debug(
                "pass",
                {
                    "reward_type": "custom_colon",
                    "custom_reward_func": f"{module_name}:score",
                    "scale": 0.75,
                },
            )
            dotted_detail = compute_multitask_reward_debug(
                "pass",
                {
                    "reward_type": "custom_dotted",
                    "custom_reward_func": f"{module_name}.score",
                    "scale": 0.5,
                },
            )

        self.assertEqual(colon_detail["reward"], 0.75)
        self.assertEqual(dotted_detail["reward"], 0.5)
        self.assertTrue(colon_detail["custom_reward"])
        self.assertTrue(dotted_detail["custom_reward"])
        self.assertEqual(calls, [("pass", 0.75), ("pass", 0.5)])

    @mock.patch("helper.code_verifier.subprocess.run")
    def test_static_code_reward_never_executes_generated_code(self, run):
        detail = compute_multitask_reward_debug(
            """```python
def solve():
    return 1
```""",
            {
                "reward_type": "code",
                "language": "python",
                "entry_point": "solve",
                "expected_substrings": ["return 1"],
            },
        )

        self.assertEqual(detail["reward"], 1.0)
        self.assertTrue(detail["passed"])
        run.assert_not_called()

    def test_plain_test_function_is_discovered_and_executed(self):
        detail = self._code_detail(
            """def add(a, b):
    return a + b""",
            """def test_add():
    assert add(2, 3) == 5""",
            entry_point="add",
        )

        self.assertEqual(detail["reward"], 1.0)
        self.assertTrue(detail["passed"])
        self.assertEqual(detail["error_type"], "none")

    def test_starter_code_is_combined_with_generated_candidate(self):
        detail = self._code_detail(
            """def solve(value):
    return helper(value)""",
            "assert solve(3) == 6",
            entry_point="solve",
            starter_code="""def helper(value):
    return value * 2""",
        )

        self.assertEqual(detail["reward"], 1.0)
        self.assertTrue(detail["passed"])
        self.assertEqual(detail["error_type"], "none")

    def test_empty_completion_is_classified_without_starting_a_process(self):
        with mock.patch("helper.code_verifier.subprocess.run") as run:
            detail = self._code_detail(
                "",
                "assert solve() == 1",
                entry_point="solve",
            )

        self.assertEqual(detail["reward"], 0.0)
        self.assertFalse(detail["passed"])
        self.assertEqual(detail["error_type"], "empty_completion")
        run.assert_not_called()

    def test_subprocess_error_classifications(self):
        cases = (
            (
                "syntax",
                "def solve(:\n    pass",
                "assert solve() == 1",
            ),
            (
                "runtime",
                "def solve():\n    raise RuntimeError('boom')",
                "solve()",
            ),
            (
                "import",
                "def solve():\n    import _definitely_missing_pros_test_package\n    return 1",
                "assert solve() == 1",
            ),
            (
                "assertion",
                "def solve():\n    return 2",
                "assert solve() == 1",
            ),
        )

        for expected_error, completion, tests in cases:
            with self.subTest(expected_error=expected_error):
                detail = self._code_detail(
                    completion,
                    tests,
                    entry_point="solve",
                )
                self.assertEqual(detail["reward"], 0.0)
                self.assertFalse(detail["passed"])
                self.assertEqual(detail["error_type"], expected_error)

    @mock.patch("helper.code_verifier.subprocess.run")
    def test_timeout_is_classified_and_bounded(self, run):
        run.side_effect = subprocess.TimeoutExpired(
            cmd=[sys.executable, "test_runner.py"],
            timeout=0.05,
            output="partial output",
            stderr="still running",
        )

        detail = self._code_detail(
            "def solve():\n    return 1",
            "assert solve() == 1",
            entry_point="solve",
            timeout_seconds=0.05,
        )

        self.assertEqual(detail["reward"], 0.0)
        self.assertFalse(detail["passed"])
        self.assertTrue(detail["timed_out"])
        self.assertEqual(detail["error_type"], "timeout")
        self.assertIn("partial output", detail["stdout_excerpt"])
        self.assertIn("still running", detail["stderr_excerpt"])
        run.assert_called_once()

    @mock.patch("helper.code_verifier.subprocess.run")
    def test_unsupported_language_is_rejected_without_execution(self, run):
        detail = self._code_detail(
            "function solve() { return 1; }",
            "assert solve() == 1",
            entry_point="solve",
            language="javascript",
        )

        self.assertEqual(detail["reward"], 0.0)
        self.assertFalse(detail["passed"])
        self.assertEqual(detail["error_type"], "unsupported_language")
        run.assert_not_called()

    @staticmethod
    def _code_detail(completion, tests, **overrides):
        example = {
            "reward_type": "code_unit_test",
            "language": "python",
            "test_type": "unit",
            "timeout_seconds": 1.0,
            "tests": tests,
            **overrides,
        }
        return compute_multitask_reward_debug(completion, example)


if __name__ == "__main__":
    unittest.main()
