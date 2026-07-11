import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastgpro_based_pros import pros_logging
from fastgpro_based_pros.pros_logging import (
    ProsEventLogger,
    RewardDebugAggregator,
    compute_generation_perf,
    compute_length_metrics,
    safe_div,
    to_json_safe,
)


class _ScalarLike:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _ArrayLike:
    def item(self):
        raise ValueError("not a scalar")

    def tolist(self):
        return [1.0, float("nan"), _ScalarLike(3)]


class _FakeSummaryWriter:
    instances = []

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.scalars = []
        self.flush_count = 0
        self.closed = False
        type(self).instances.append(self)

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


class JsonAndMetricHelpersTest(unittest.TestCase):
    def test_json_safe_duck_types_and_nonfinite_values(self):
        value = {
            "scalar": _ScalarLike(7),
            "array": _ArrayLike(),
            "tuple": (float("inf"), Path("a/b")),
        }
        converted = to_json_safe(value)
        self.assertEqual(converted["scalar"], 7)
        self.assertEqual(converted["array"], [1.0, 0.0, 3])
        self.assertEqual(converted["tuple"], [0.0, "a/b"])
        json.dumps(converted, allow_nan=False)

    def test_safe_div_never_returns_nonfinite(self):
        self.assertEqual(safe_div(5, 0), 0.0)
        self.assertEqual(safe_div(float("inf"), 2), 0.0)
        self.assertEqual(safe_div(4, 2), 2.0)
        self.assertEqual(safe_div(5, 0, default=3), 3.0)

    def test_length_metrics_use_sample_standard_deviation(self):
        metrics = compute_length_metrics([2, 4], [1, 1], [3, 5])
        self.assertEqual(metrics["suffix_token_length_count"], 2)
        self.assertEqual(metrics["suffix_token_length_mean"], 3.0)
        self.assertEqual(metrics["suffix_token_length_min"], 2.0)
        self.assertEqual(metrics["suffix_token_length_max"], 4.0)
        self.assertAlmostEqual(metrics["suffix_token_length_stdev"], math.sqrt(2.0))
        self.assertEqual(metrics["suffix_token_length_range"], 2.0)
        self.assertAlmostEqual(metrics["suffix_token_length_cv"], math.sqrt(2.0) / 3.0)
        self.assertEqual(metrics["inherited_partial_rollout_length_stdev"], 0.0)
        self.assertEqual(metrics["full_response_length_mean"], 4.0)

        empty = compute_length_metrics([], [], [])
        self.assertEqual(empty["suffix_token_length_count"], 0)
        self.assertEqual(empty["suffix_token_length_stdev"], 0.0)
        json.dumps(empty, allow_nan=False)

    def test_speculative_generation_perf_formulas(self):
        outputs = {
            "total_time_cost": 5.0,
            "target_time_cost": 2.0,
            "draft_time_cost": 1.0,
            "check_time_cost": 0.5,
            "prefill_time_cost": 0.25,
            "speculative_emitted_tokens": 12,
            "speculative_accepted_draft_tokens": 6,
            "speculative_verified_draft_tokens": 10,
            "speculative_path_budget_tokens": 8,
            "speculative_verification_rounds": 4,
        }
        metrics = compute_generation_perf(
            outputs,
            [3, 5],
            generation_time_sec=4.0,
            generation_backend="speculative",
        )
        self.assertEqual(metrics["generated_completion_tokens"], 8)
        self.assertEqual(metrics["generated_tokens_per_second"], 2.0)
        self.assertEqual(metrics["reported_generation_time_sec"], 5.0)
        self.assertEqual(metrics["speculative_avg_emitted_tokens_per_round"], 3.0)
        self.assertEqual(metrics["speculative_avg_accepted_draft_tokens_per_round"], 1.5)
        self.assertEqual(metrics["speculative_path_acceptance_rate"], 0.75)
        self.assertEqual(metrics["speculative_tree_acceptance_rate"], 0.6)
        self.assertEqual(metrics["speculative_verified_draft_tokens_per_round"], 2.5)
        self.assertEqual(metrics["target_time_ratio"], 0.5)
        self.assertEqual(metrics["draft_time_ratio"], 0.25)
        self.assertEqual(metrics["check_time_ratio"], 0.125)
        json.dumps(metrics, allow_nan=False)

    def test_target_generation_has_same_schema_and_zero_speculative_metrics(self):
        outputs = {
            "generated_token_ids": [[1, 2], [3]],
            "total_time_cost": 0,
            "speculative_emitted_tokens": 999,
            "speculative_accepted_draft_tokens": 999,
            "speculative_verified_draft_tokens": 999,
            "speculative_path_budget_tokens": 999,
            "speculative_verification_rounds": 999,
        }
        target = compute_generation_perf(outputs, generation_backend="target")
        speculative = compute_generation_perf({}, [], generation_backend="speculative")
        speculative_keys = [key for key in target if key.startswith("speculative_")]
        self.assertTrue(speculative_keys)
        for key in speculative_keys:
            self.assertEqual(target[key], 0, key)
            self.assertIn(key, speculative)
        self.assertEqual(target["generated_completion_tokens"], 3)
        self.assertEqual(target["generated_tokens_per_second"], 0.0)
        json.dumps(target, allow_nan=False)


class RewardDebugAggregatorTest(unittest.TestCase):
    def test_batch_cumulative_per_task_and_group_decisions(self):
        aggregator = RewardDebugAggregator(sample_limit=2, char_limit=40)
        passing = {
            "reward": 1.0,
            "passed": True,
            "reward_type": "exact_match",
            "error_type": "none",
            "answer_parsed": True,
            "gold_parsed": True,
            "format_reward": 1.0,
        }
        failing = {
            "reward": 0.0,
            "passed": False,
            "reward_type": "code_unit_test",
            "error_type": "timeout",
            "timed_out": True,
            "has_tests": False,
            "has_entry_point": False,
            "test_type": "assert",
            "stderr_excerpt": "e" * 200,
        }
        aggregator.record_completion(
            passing,
            task_id="math",
            prompt=[{"role": "user", "content": "2+2?"}],
            completion="4",
            repeat_index=0,
        )
        aggregator.record_completion(
            failing,
            task_id="code",
            prompt="write a function " * 20,
            completion="bad code " * 20,
            repeat_index=1,
        )
        aggregator.record_group_decision("used", [passing], task_id="math")
        aggregator.record_group_decision("ignore_due_incorrect", [failing], task_id="code")

        snapshot = aggregator.snapshot()
        batch = snapshot["batch"]
        self.assertEqual(batch["completion_count"], 2)
        self.assertEqual(batch["mean_reward_all_completions"], 0.5)
        self.assertEqual(batch["reward_std_all_completions"], 0.5)
        self.assertEqual(batch["pass_count"], 1)
        self.assertEqual(batch["fail_count"], 1)
        self.assertEqual(batch["pass_rate"], 0.5)
        self.assertEqual(batch["timeout_count"], 1)
        self.assertEqual(batch["missing_tests_count"], 1)
        self.assertEqual(batch["missing_entry_point_count"], 1)
        self.assertEqual(batch["used_group_count"], 1)
        self.assertEqual(batch["skip_due_incorrect_group_count"], 1)
        self.assertEqual(batch["reward_type_counts"], {"code_unit_test": 1, "exact_match": 1})
        self.assertEqual(batch["ignored_incorrect_error_type_counts"], {"timeout": 1})
        self.assertEqual(snapshot["per_task_batch"]["math"]["used_group_count"], 1)
        self.assertEqual(snapshot["per_task_batch"]["code"]["timeout_count"], 1)
        self.assertEqual(len(snapshot["failed_samples"]), 1)
        self.assertLessEqual(len(snapshot["failed_samples"][0]["prompt"]), 40)
        self.assertLessEqual(len(snapshot["failed_samples"][0]["completion"]), 40)
        self.assertLessEqual(len(snapshot["failed_samples"][0]["stderr_excerpt"]), 40)
        json.dumps(snapshot, allow_nan=False)

        payload = aggregator.as_payload()
        self.assertIn("reward_debug_batch", payload)
        self.assertIn("reward_debug_per_task", payload)
        aggregator.reset_batch()
        after_reset = aggregator.snapshot()
        self.assertEqual(after_reset["batch"]["completion_count"], 0)
        self.assertEqual(after_reset["cumulative"]["completion_count"], 2)
        self.assertEqual(after_reset["per_task"]["code"]["skip_group_count"], 1)

    def test_failed_samples_are_capped_and_stdio_does_not_require_entry_point(self):
        aggregator = RewardDebugAggregator(sample_limit=1, char_limit=16)
        stdio_failure = {
            "reward": 0,
            "passed": False,
            "reward_type": "code_unit_test",
            "error_type": "wrong_answer",
            "has_tests": True,
            "has_entry_point": False,
            "test_type": "stdin_stdout",
        }
        aggregator.record_completion(stdio_failure, task_id="code", prompt="p" * 30, completion="c" * 30)
        aggregator.record_completion(stdio_failure, task_id="code", prompt="second", completion="second")
        snapshot = aggregator.snapshot()
        self.assertEqual(snapshot["batch"]["missing_entry_point_count"], 0)
        self.assertEqual(len(snapshot["failed_samples"]), 1)
        self.assertLessEqual(len(snapshot["failed_samples"][0]["prompt"]), 16)

    def test_invalid_group_decision_is_rejected(self):
        aggregator = RewardDebugAggregator()
        with self.assertRaisesRegex(ValueError, "Unknown rollout group decision"):
            aggregator.record_group_decision("mystery")


class ProsEventLoggerTest(unittest.TestCase):
    def setUp(self):
        _FakeSummaryWriter.instances.clear()

    def test_jsonl_event_order_tensorboard_filtering_flush_and_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "train.jsonl"
            writer = _FakeSummaryWriter()
            logger = ProsEventLogger(path, summary_writer=writer)
            generation = logger.log(
                "generation",
                {
                    "tb_step": 99,
                    "epoch": 1,
                    "bad_value": float("nan"),
                    "dataset": {"tree nodes": 3},
                    "task_metrics": {"code task": {"reward": 0.5}},
                    "large_values": [1, 2, 3],
                    "text": "do not write this to TensorBoard",
                },
                global_step=0,
                generation_attempt=1,
                generation_backend="target",
            )
            train = logger.log("train", {"actor/loss": 0.25}, global_step=1)

            self.assertEqual(generation["tb_step"], 0)
            self.assertEqual(train["tb_step"], 1)
            self.assertEqual(logger.tb_step, 2)
            self.assertIn("timestamp", generation)
            self.assertIn("elapsed_time_sec", generation)

            tags = {tag for tag, _value, _step in writer.scalars}
            self.assertIn("generation/dataset/tree_nodes", tags)
            self.assertIn("generation/task_metrics/code_task/reward", tags)
            self.assertIn("train/actor/loss", tags)
            self.assertFalse(any("large_values" in tag for tag in tags))
            self.assertFalse(any(tag.endswith("/text") for tag in tags))
            for _tag, value, step in writer.scalars:
                self.assertTrue(math.isfinite(value))
                self.assertIn(step, {0, 1})

            logger.close()
            self.assertTrue(writer.closed)
            self.assertGreaterEqual(writer.flush_count, 3)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                logger.log("generation", {})

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["event"] for row in rows], ["generation", "train"])
            self.assertEqual([row["tb_step"] for row in rows], [0, 1])
            self.assertEqual(rows[0]["bad_value"], 0.0)
            for row in rows:
                json.dumps(row, allow_nan=False)

    def test_default_tensorboard_directory_and_injected_writer_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run" / "events.jsonl"
            with ProsEventLogger(path, summary_writer_cls=_FakeSummaryWriter) as logger:
                self.assertIsInstance(logger.writer, _FakeSummaryWriter)
                expected = path.parent / "tensorboard"
                self.assertEqual(Path(logger.writer.log_dir), expected)
                logger.log("config", {"dataset/examples": 4})
            self.assertTrue(_FakeSummaryWriter.instances[-1].closed)

    def test_tensorboard_unavailable_warns_and_jsonl_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            warning_messages = []
            with mock.patch.object(
                pros_logging,
                "_load_summary_writer_class",
                side_effect=ModuleNotFoundError("tensorboard"),
            ):
                with ProsEventLogger(path, warning_fn=warning_messages.append) as logger:
                    self.assertIsNone(logger.writer)
                    logger.log("generation", {"reward": 1.0})
            self.assertEqual(len(warning_messages), 1)
            self.assertIn("continuing with JSONL", warning_messages[0])
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["event"], "generation")


if __name__ == "__main__":
    unittest.main()
