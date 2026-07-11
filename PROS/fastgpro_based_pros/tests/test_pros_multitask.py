import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "FastGRPO"))


class ProsMultiTaskLoaderTest(unittest.TestCase):
    def test_inline_math_qa_and_code_normalization_is_seeded_and_nested(self):
        from helper.multitask import load_multitask_QAs, render_messages

        config = {
            "tasks": [
                {
                    "id": "math",
                    "prompt_type": "math",
                    "prompt_field": ["missing", "payload.problem"],
                    "answer_field": ["missing", "labels.answer"],
                    "records": [
                        {"payload": {"problem": "2 + 2?"}, "labels": {"answer": "4"}},
                        {"payload": {"problem": "3 + 3?"}, "labels": {"answer": "6"}},
                    ],
                },
                {
                    "id": "qa",
                    "prompt_type": "qa",
                    "reward_type": "exact_match",
                    "messages_field": ["missing", "chat.messages"],
                    "answer_field": "label.value",
                    "records": [
                        {
                            "chat": {"messages": [{"role": "user", "content": "Capital of France?"}]},
                            "label": {"value": "Paris"},
                        }
                    ],
                },
                {
                    "id": "code",
                    "prompt_type": "code",
                    "reward_type": "code_unit_test",
                    "prompt_field": "spec.prompt",
                    "entry_point_field": "spec.entry",
                    "tests_field": "checks.tests",
                    "metadata_fields": ["difficulty", "source.name"],
                    "records": [
                        {
                            "spec": {"prompt": "Return one.", "entry": "solve"},
                            "checks": {"tests": "assert solve() == 1"},
                            "difficulty": "easy",
                            "source": {"name": "inline"},
                        }
                    ],
                },
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
            json.dump(config, handle)
            handle.flush()
            first = load_multitask_QAs(handle.name, seed=123)
            second = load_multitask_QAs(handle.name, seed=123)

        self.assertEqual(first, second)
        self.assertEqual({sample["task_id"] for sample in first}, {"math", "qa", "code"})
        math = next(sample for sample in first if sample["task_id"] == "math" and "2 + 2" in sample["question"])
        qa = next(sample for sample in first if sample["task_id"] == "qa")
        code = next(sample for sample in first if sample["task_id"] == "code")
        self.assertEqual((math["question"], math["answer"]), ("2 + 2?", "4"))
        self.assertEqual(render_messages(qa), [{"role": "user", "content": "Capital of France?"}])
        self.assertEqual((code["entry_point"], code["tests"]), ("solve", "assert solve() == 1"))
        self.assertEqual(code["metadata"]["difficulty"], "easy")


class ProsMultiTaskTreeTest(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("numpy is not installed")

    @staticmethod
    def _classes():
        from fastgpro_based_pros.pros_tree import ProsTreeConfig, ProsTreeEngine

        return ProsTreeConfig, ProsTreeEngine

    def _weighted_engine(self, task_ids, weights, seed=7):
        ProsTreeConfig, ProsTreeEngine = self._classes()
        return ProsTreeEngine(
            original_data_len=len(task_ids),
            config=ProsTreeConfig(random_seed=seed),
            root_task_ids=task_ids,
            task_weights=weights,
        )

    @staticmethod
    def _task_counts(engine, batch):
        return Counter(engine.get_task_id(item) for item in batch)

    def test_largest_remainder_weights_produce_five_to_three_batch(self):
        engine = self._weighted_engine(
            task_ids=["math"] * 10 + ["code"] * 6,
            weights={"math": 5.0, "code": 3.0},
        )

        batch, metrics = engine.select_batch(batch_size=8, step_num=0)

        self.assertEqual(self._task_counts(engine, batch), {"math": 5, "code": 3})
        self.assertEqual(metrics["sampler/task_quotas"], {"math": 5, "code": 3})
        self.assertEqual(metrics["sampler/task_selected_counts"], {"math": 5, "code": 3})
        self.assertEqual(metrics["sampler/task_quota_shortfall"], 0.0)

    def test_child_inherits_task_and_quota_preserves_parent_diversity(self):
        engine = self._weighted_engine(
            task_ids=["math"] * 10 + ["code"] * 6,
            weights={"math": 5.0, "code": 3.0},
        )
        child = engine.create_new_node(
            father_node=engine.get_node(0),
            partial_rollout=[10, 11, 12],
            step_num=1,
            score=1.0,
        )
        # Make the child maximally uncertain so the task-aware selector should
        # prefer it over its root while still selecting that ancestor once.
        engine.psi[:] = 4.0
        engine.psi[child] = 0.0

        batch, _ = engine.select_batch(batch_size=8, step_num=1)
        ancestors = [engine.get_original_ancestor_item(item) for item in batch]

        self.assertEqual(engine.get_task_id(child), "math")
        self.assertIn(child, batch)
        self.assertEqual(self._task_counts(engine, batch), {"math": 5, "code": 3})
        self.assertEqual(len(ancestors), len(set(ancestors)))

    def test_multi_task_without_explicit_weights_retains_legacy_selection(self):
        ProsTreeConfig, ProsTreeEngine = self._classes()
        config = ProsTreeConfig(random_seed=19)
        legacy = ProsTreeEngine(original_data_len=8, config=config)
        tagged_unweighted = ProsTreeEngine(
            original_data_len=8,
            config=ProsTreeConfig(random_seed=19),
            root_task_ids=["math"] * 4 + ["code"] * 4,
        )

        legacy_batch, legacy_metrics = legacy.select_batch(batch_size=4, step_num=0)
        tagged_batch, tagged_metrics = tagged_unweighted.select_batch(batch_size=4, step_num=0)

        self.assertFalse(tagged_unweighted.task_weighting_enabled)
        self.assertEqual(tagged_batch, legacy_batch)
        self.assertEqual(tagged_metrics, legacy_metrics)

    def test_explicit_single_task_weights_retain_legacy_selection(self):
        _, ProsTreeEngine = self._classes()
        from fastgpro_based_pros.pros_tree import ProsTreeConfig

        legacy = ProsTreeEngine(8, ProsTreeConfig(random_seed=23))
        weighted_single = ProsTreeEngine(
            8,
            ProsTreeConfig(random_seed=23),
            root_task_ids=["math"] * 8,
            task_weights={"math": 9.0},
        )

        self.assertFalse(weighted_single.task_weighting_enabled)
        self.assertEqual(
            weighted_single.select_batch(batch_size=4, step_num=0),
            legacy.select_batch(batch_size=4, step_num=0),
        )

    def test_zero_weight_task_is_never_selected(self):
        engine = self._weighted_engine(
            task_ids=["math"] * 5 + ["code"] * 5,
            weights={"math": 1.0, "code": 0.0},
        )

        batch, metrics = engine.select_batch(batch_size=5, step_num=0)

        self.assertEqual(self._task_counts(engine, batch), {"math": 5})
        self.assertEqual(metrics["sampler/task_quotas"], {"math": 5, "code": 0})
        self.assertEqual(metrics["sampler/task_selected_counts"]["code"], 0)
        self.assertEqual(metrics["sampler/task_fallback_counts"]["code"], 0)

    def test_all_zero_and_negative_weights_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "sum to a positive"):
            self._weighted_engine(
                task_ids=["math", "code"],
                weights={"math": 0.0, "code": 0.0},
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._weighted_engine(
                task_ids=["math", "code"],
                weights={"math": 1.0, "code": -1.0},
            )

    def test_task_scarcity_is_backfilled_and_reported(self):
        engine = self._weighted_engine(
            task_ids=["math"] * 5 + ["code"] * 2,
            weights={"math": 1.0, "code": 1.0},
        )

        batch, metrics = engine.select_batch(batch_size=6, step_num=0)

        self.assertEqual(self._task_counts(engine, batch), {"math": 4, "code": 2})
        self.assertEqual(metrics["sampler/task_quotas"], {"math": 3, "code": 3})
        self.assertEqual(metrics["sampler/task_shortfall_counts"], {"math": 0, "code": 1})
        self.assertEqual(metrics["sampler/task_fallback_counts"], {"math": 1, "code": 0})
        self.assertEqual(metrics["sampler/task_quota_shortfall"], 1.0)
        self.assertEqual(metrics["sampler/task_fallback_fill"], 1.0)
        ancestors = [engine.get_original_ancestor_item(item) for item in batch]
        self.assertEqual(len(ancestors), len(set(ancestors)))

    def test_checkpoint_task_metadata_round_trip_and_legacy_load(self):
        weighted = self._weighted_engine(
            task_ids=["math"] * 4 + ["code"] * 4,
            weights={"math": 3.0, "code": 1.0},
        )
        state = weighted.state_dict()

        _, ProsTreeEngine = self._classes()
        from fastgpro_based_pros.pros_tree import ProsTreeConfig

        restored = ProsTreeEngine(8, ProsTreeConfig(random_seed=9))
        restored.load_state_dict(state)
        self.assertTrue(restored.task_weighting_enabled)
        self.assertEqual(restored.root_task_ids, weighted.root_task_ids)
        self.assertEqual(restored.task_weights, weighted.task_weights)

        legacy_state = dict(state)
        legacy_state.pop("root_task_ids")
        legacy_state.pop("task_weights")
        configured = ProsTreeEngine(
            8,
            ProsTreeConfig(random_seed=9),
            root_task_ids=["qa"] * 8,
            task_weights={"qa": 1.0},
        )
        configured.load_state_dict(legacy_state)
        self.assertEqual(configured.root_task_ids, ["qa"] * 8)
        self.assertEqual(configured.task_weights, {"qa": 1.0})


if __name__ == "__main__":
    unittest.main()
