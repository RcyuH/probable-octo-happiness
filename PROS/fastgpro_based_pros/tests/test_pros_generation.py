import sys
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class _CudaStub:
    @staticmethod
    def is_available():
        return False


class _TorchStub:
    cuda = _CudaStub()

    @staticmethod
    @contextmanager
    def inference_mode():
        yield

    @staticmethod
    @contextmanager
    def no_grad():
        yield


class _ModeSpy:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


class _TrainModeSpy:
    def __init__(self):
        self.train_calls = 0

    def train(self):
        self.train_calls += 1
        return self


class _AmbiguousEmptySequence:
    def __bool__(self):
        raise RuntimeError("ambiguous truth value")

    def __len__(self):
        return 0


class _RewardDebugSpy:
    def __init__(self):
        self.completions = []
        self.decisions = []
        self.reset_calls = 0

    def reset_batch(self):
        self.reset_calls += 1

    def record_completion(self, detail, **context):
        self.completions.append((dict(detail), context))

    def record_group_decision(self, decision, details, **context):
        self.decisions.append((decision, list(details), context))


class _TokenizerStub:
    pad_token_id = 0
    eos_token_id = 2

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token) for token in token_ids)


class ProsGenerationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
        except ModuleNotFoundError:
            raise unittest.SkipTest("numpy is required to import pros_trainer")

    @staticmethod
    def _imports():
        from fastgpro_based_pros.pros_config import ProsConfig
        from fastgpro_based_pros.pros_trainer import EncodedPrompt, ProsTrainer, RolloutBatch, TrainingRecord

        return ProsConfig, EncodedPrompt, ProsTrainer, RolloutBatch, TrainingRecord

    def _bare_trainer(self, **config_updates):
        ProsConfig, _, ProsTrainer, _, _ = self._imports()
        defaults = {
            "batch_size": 1,
            "repeated_generate_nums": 2,
            "verification_capacity": 32,
            "train_draft": False,
            "use_tensorboard": False,
        }
        defaults.update(config_updates)
        trainer = object.__new__(ProsTrainer)
        trainer.cfg = ProsConfig(**defaults)
        trainer.torch = _TorchStub()
        trainer.device = "cpu"
        trainer.tokenizer = _TokenizerStub()
        trainer.reward_debug = _RewardDebugSpy()
        trainer.task_stats = {}
        trainer.used_groups = 0
        trainer.render_messages = lambda example: example.get("messages", [example.get("question", "")])
        return trainer

    @staticmethod
    def _prompt(*, item=0, partial=None, example=None):
        _, EncodedPrompt, _, _, _ = ProsGenerationIntegrationTest._imports()
        partial = list(partial or [])
        example = dict(example or {"question": "q", "task_id": "math", "reward_type": "zero"})
        return EncodedPrompt(
            item=item,
            ancestor_item=item,
            question=str(example.get("question", "q")),
            answer=example.get("answer"),
            example=example,
            task_id=str(example.get("task_id", "default")),
            prompt_ids=[10, 11],
            partial_rollout=partial,
            input_ids=[10, 11] + partial,
        )

    def test_training_speculative_generation_forwards_all_six_controls_and_sets_eval_mode(self):
        trainer = self._bare_trainer(
            generation_backend="speculative",
            verification_capacity=42,
            max_draft_token_length=7,
            min_draft_token_length=2,
            max_draft_k=11,
            max_verification_num=123,
            draft_token_length_c=0.625,
        )
        target_model = _ModeSpy()
        draft_model = _ModeSpy()
        trainer.model = SimpleNamespace(target_model=target_model, draft_model=draft_model)
        trainer._pad_left = mock.Mock(return_value=("input_ids", "attention_mask"))
        trainer.speculative_generate = mock.Mock(return_value={"generated_token_ids": [[1], [2]]})

        result = trainer._generate([self._prompt()])

        self.assertEqual(result["generated_token_ids"], [[1], [2]])
        self.assertEqual(target_model.eval_calls, 1)
        self.assertEqual(draft_model.eval_calls, 1)
        kwargs = trainer.speculative_generate.call_args.kwargs
        self.assertEqual(
            {name: kwargs[name] for name in (
                "verification_capacity",
                "max_draft_token_length",
                "min_draft_token_length",
                "max_draft_k",
                "max_verification_num",
                "draft_token_length_c",
            )},
            {
                "verification_capacity": 42,
                "max_draft_token_length": 7,
                "min_draft_token_length": 2,
                "max_draft_k": 11,
                "max_verification_num": 123,
                "draft_token_length_c": 0.625,
            },
        )

    def test_draft_training_generation_uses_no_grad_not_inference_tensors(self):
        trainer = self._bare_trainer(generation_backend="speculative", train_draft=True)
        trainer.model = SimpleNamespace(target_model=_ModeSpy(), draft_model=_ModeSpy())
        trainer._pad_left = mock.Mock(return_value=("input_ids", "attention_mask"))
        trainer.speculative_generate = mock.Mock(return_value={"generated_token_ids": [[1], [2]]})
        no_grad = mock.Mock(side_effect=lambda: nullcontext())
        inference_mode = mock.Mock(side_effect=lambda: nullcontext())
        trainer.torch = SimpleNamespace(
            cuda=_CudaStub(),
            no_grad=no_grad,
            inference_mode=inference_mode,
        )

        trainer._generate([self._prompt()])

        no_grad.assert_called_once_with()
        inference_mode.assert_not_called()

    def test_evaluation_forwards_all_six_controls_and_calls_debug_reward_once(self):
        trainer = self._bare_trainer(
            generation_backend="speculative",
            repeated_generate_nums=1,
            verification_capacity=19,
            max_draft_token_length=6,
            min_draft_token_length=1,
            max_draft_k=9,
            max_verification_num=77,
            draft_token_length_c=0.5,
            eval_samples=1,
        )
        target_model = _ModeSpy()
        draft_model = _ModeSpy()
        trainer.model = SimpleNamespace(target_model=target_model, draft_model=draft_model)
        trainer.eval_qas = [{"question": "2+2?", "task_id": "math", "reward_type": "zero"}]
        trainer._encode_base_prompt = mock.Mock(return_value=[10, 11])
        trainer._pad_left = mock.Mock(return_value=("eval_input_ids", "eval_attention_mask"))
        trainer.speculative_generate = mock.Mock(return_value={"generated_token_ids": [[30, 31]]})
        trainer.compute_multitask_reward_debug = mock.Mock(
            return_value={"reward": 0.75, "passed": True, "reward_type": "zero", "error_type": "none"}
        )

        metrics = trainer._evaluate()

        self.assertEqual(metrics["eval/samples"], 1.0)
        self.assertEqual(metrics["eval/reward_mean"], 0.75)
        self.assertEqual(trainer.compute_multitask_reward_debug.call_count, 1)
        self.assertEqual(target_model.eval_calls, 1)
        self.assertEqual(draft_model.eval_calls, 1)
        kwargs = trainer.speculative_generate.call_args.kwargs
        self.assertEqual(
            [
                kwargs["verification_capacity"],
                kwargs["max_draft_token_length"],
                kwargs["min_draft_token_length"],
                kwargs["max_draft_k"],
                kwargs["max_verification_num"],
                kwargs["draft_token_length_c"],
            ],
            [19, 6, 1, 9, 77, 0.5],
        )

    def test_training_output_count_mismatch_fails_before_reward_execution(self):
        trainer = self._bare_trainer(repeated_generate_nums=2)
        trainer.compute_multitask_reward_debug = mock.Mock(return_value={"reward": 0.0})

        with self.assertRaisesRegex(ValueError, r"expected 2.*got 1"):
            trainer._build_training_records(
                [self._prompt()],
                {"generated_token_ids": [[30]]},
            )

        trainer.compute_multitask_reward_debug.assert_not_called()

    def test_evaluation_output_count_mismatch_is_reported(self):
        trainer = self._bare_trainer(
            generation_backend="speculative",
            repeated_generate_nums=1,
            eval_samples=1,
        )
        trainer.model = SimpleNamespace(target_model=_ModeSpy(), draft_model=_ModeSpy())
        trainer.eval_qas = [{"question": "q", "task_id": "qa", "reward_type": "zero"}]
        trainer._encode_base_prompt = mock.Mock(return_value=[10])
        trainer._pad_left = mock.Mock(return_value=("ids", "mask"))
        trainer.speculative_generate = mock.Mock(return_value={"generated_token_ids": []})
        trainer.compute_multitask_reward_debug = mock.Mock(return_value={"reward": 0.0})

        with self.assertRaisesRegex(ValueError, r"during evaluation.*expected 1, got 0"):
            trainer._evaluate()

        trainer.compute_multitask_reward_debug.assert_not_called()

    def test_full_partial_plus_suffix_is_rewarded_once_and_only_suffix_is_trainable(self):
        trainer = self._bare_trainer(repeated_generate_nums=2, drop_zero_std_groups=True)
        trainer.compute_multitask_reward_debug = mock.Mock(
            side_effect=[
                {"reward": 0.0, "passed": False, "reward_type": "exact_match", "error_type": "wrong_answer"},
                {"reward": 1.0, "passed": True, "reward_type": "exact_match", "error_type": "none"},
            ]
        )
        prompt = self._prompt(partial=[20, 21])

        batch = trainer._build_training_records(
            [prompt],
            {"generated_token_ids": [[30, 31], [40]]},
        )

        self.assertEqual(trainer.compute_multitask_reward_debug.call_count, 2)
        rewarded_text = [call.args[0] for call in trainer.compute_multitask_reward_debug.call_args_list]
        self.assertEqual(rewarded_text, ["20 21 30 31", "20 21 40"])
        self.assertEqual(len(batch.all_records), 2)
        self.assertEqual(len(batch.actor_records), 2)
        self.assertEqual(batch.all_records[0].full_response_ids, [20, 21, 30, 31])
        self.assertEqual(batch.all_records[0].full_input_ids, [10, 11, 20, 21, 30, 31])
        self.assertEqual(batch.all_records[0].new_token_mask, [0, 0, 0, 0, 1, 1])
        self.assertEqual(batch.all_records[1].new_token_mask, [0, 0, 0, 0, 1])
        self.assertEqual(len(trainer.reward_debug.completions), 2)

    def test_zero_std_group_remains_in_all_records_but_not_actor_records(self):
        trainer = self._bare_trainer(repeated_generate_nums=2, drop_zero_std_groups=True)
        trainer.compute_multitask_reward_debug = mock.Mock(
            return_value={"reward": 1.0, "passed": True, "reward_type": "exact_match", "error_type": "none"}
        )

        batch = trainer._build_training_records(
            [self._prompt()],
            {"generated_token_ids": [[30], [31]]},
        )

        self.assertEqual(len(batch.all_records), 2)
        self.assertEqual(batch.actor_records, [])
        self.assertEqual(batch.group_decisions, {0: "ignore_due_correct"})
        self.assertEqual(batch.skipped_correct_groups, 1)
        self.assertEqual(trainer.compute_multitask_reward_debug.call_count, 2)

    def test_all_empty_group_remains_visible_to_tree_but_not_actor(self):
        trainer = self._bare_trainer(repeated_generate_nums=2, drop_zero_std_groups=True)
        trainer.compute_multitask_reward_debug = mock.Mock(
            return_value={"reward": 0.0, "passed": False, "reward_type": "zero", "error_type": "empty_completion"}
        )

        batch = trainer._build_training_records(
            [self._prompt()],
            {"generated_token_ids": [[], []]},
        )

        self.assertEqual(len(batch.all_records), 2)
        self.assertEqual(batch.actor_records, [])
        self.assertEqual(batch.empty_completions, 2)
        self.assertEqual(batch.group_decisions, {0: "ignore_due_incorrect"})
        self.assertEqual(batch.skipped_incorrect_groups, 1)
        self.assertEqual(trainer.compute_multitask_reward_debug.call_count, 2)

    def test_passing_partial_with_empty_suffix_is_classified_as_correct_skip(self):
        trainer = self._bare_trainer(repeated_generate_nums=2, drop_zero_std_groups=True)
        trainer.compute_multitask_reward_debug = mock.Mock(
            return_value={"reward": 1.0, "passed": True, "reward_type": "exact_match", "error_type": "none"}
        )

        batch = trainer._build_training_records(
            [self._prompt(partial=[20, 21])],
            {"generated_token_ids": [[], []]},
        )

        self.assertEqual(len(batch.all_records), 2)
        self.assertEqual(batch.actor_records, [])
        self.assertEqual(batch.group_decisions, {0: "ignore_due_correct"})
        self.assertEqual(batch.skipped_correct_groups, 1)
        self.assertEqual(batch.skipped_incorrect_groups, 0)

    def test_mixed_empty_and_nonempty_group_excludes_empty_suffix_from_actor(self):
        trainer = self._bare_trainer(repeated_generate_nums=2, drop_zero_std_groups=True)
        trainer.compute_multitask_reward_debug = mock.Mock(
            side_effect=[
                {"reward": 0.0, "passed": False, "reward_type": "exact_match", "error_type": "empty"},
                {"reward": 1.0, "passed": True, "reward_type": "exact_match", "error_type": "none"},
            ]
        )

        batch = trainer._build_training_records(
            [self._prompt()],
            {"generated_token_ids": [[], [30]]},
        )

        self.assertEqual(len(batch.all_records), 2)
        self.assertEqual([record.generated_ids for record in batch.actor_records], [[30]])
        self.assertEqual(batch.group_decisions, {0: "used"})
        self.assertEqual(batch.used_groups, 1)

    def test_target_generator_emits_zero_speculative_counters(self):
        class FakeBatch:
            shape = (1, 2)

            def to(self, device):
                del device
                return self

            def repeat_interleave(self, repeats, dim=0):
                del repeats, dim
                return self

        class FakeSequence:
            def __init__(self, values):
                self.values = list(values)

            def __getitem__(self, index):
                if isinstance(index, slice):
                    return type(self)(self.values[index])
                return self.values[index]

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self.values)

        class TargetModel(_ModeSpy):
            @staticmethod
            def generate(**kwargs):
                del kwargs
                return [FakeSequence([10, 11, 30, 2])]

        trainer = self._bare_trainer(generation_backend="target", repeated_generate_nums=1)
        trainer.model = SimpleNamespace(target_model=TargetModel())

        outputs = trainer._target_generate(
            FakeBatch(),
            FakeBatch(),
            do_sample=True,
            repeated_generate_nums=1,
            temperature=1.0,
            top_p=0.95,
            top_k=None,
            statistical_time=False,
            max_length=8,
        )

        self.assertEqual(outputs["generated_token_ids"], [[30, 2]])
        for key in (
            "speculative_emitted_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_verified_draft_tokens",
            "speculative_path_budget_tokens",
            "speculative_verification_rounds",
        ):
            self.assertEqual(outputs[key], 0, key)

    def test_target_generator_preserves_eos_when_pad_and_eos_ids_match(self):
        class FakeBatch:
            shape = (1, 2)

            def to(self, device):
                del device
                return self

            def repeat_interleave(self, repeats, dim=0):
                del repeats, dim
                return self

        class FakeSequence:
            def __getitem__(self, index):
                values = [10, 11, 30, 2, 2]
                selected = values[index]
                return type(self)(selected) if isinstance(index, slice) else selected

            def __init__(self, values=None):
                self.values = [10, 11, 30, 2, 2] if values is None else list(values)

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self.values)

        class TargetModel(_ModeSpy):
            @staticmethod
            def generate(**kwargs):
                del kwargs
                return [FakeSequence()]

        trainer = self._bare_trainer(generation_backend="target", repeated_generate_nums=1)
        trainer.tokenizer = SimpleNamespace(pad_token_id=2, eos_token_id=2)
        trainer.model = SimpleNamespace(target_model=TargetModel())

        outputs = trainer._target_generate(
            FakeBatch(),
            FakeBatch(),
            do_sample=True,
            repeated_generate_nums=1,
            temperature=1.0,
            top_p=0.95,
            top_k=None,
            statistical_time=False,
            max_length=8,
        )

        self.assertEqual(outputs["generated_token_ids"], [[30, 2]])

    def test_empty_tensor_like_draft_outputs_are_not_boolean_coerced(self):
        trainer = self._bare_trainer(generation_backend="speculative", train_draft=True)
        draft_model = _TrainModeSpy()
        trainer.model = SimpleNamespace(draft_model=draft_model)

        metrics = trainer._train_draft_model(
            {
                "all_draft_input_states": _AmbiguousEmptySequence(),
                "all_draft_input_ids": _AmbiguousEmptySequence(),
            },
            [self._prompt()],
        )

        self.assertEqual(metrics, {"draft/loss1": 0.0, "draft/loss2": 0.0})
        self.assertEqual(draft_model.train_calls, 1)

    def test_epoch_is_finite_and_updates_tree_even_when_every_group_is_actor_skipped(self):
        _, _, _, RolloutBatch, TrainingRecord = self._imports()
        trainer = self._bare_trainer(
            generation_backend="target",
            repeated_generate_nums=1,
            max_train_steps=0,
        )
        trainer.global_step = 0
        trainer.generation_attempt = 0
        trainer.next_items = [0]
        trainer.optimizer_draft = None
        prompt = self._prompt()
        record = TrainingRecord(
            item=0,
            ancestor_item=0,
            prompt_ids=[10, 11],
            partial_rollout=[],
            generated_ids=[30],
            full_response_ids=[30],
            full_input_ids=[10, 11, 30],
            new_token_mask=[0, 0, 1],
            reward=1.0,
            decoded_completion="30",
            task_id="math",
            reward_example=prompt.example,
            reward_detail={"reward": 1.0},
        )
        skipped_batch = RolloutBatch(
            all_records=[record],
            actor_records=[],
            group_decisions={0: "ignore_due_correct"},
            generated_groups=1,
            skipped_correct_groups=1,
        )
        trainer._encode_selected_items = mock.Mock(return_value=[prompt])
        trainer._generate = mock.Mock(return_value={"generated_token_ids": [[30]]})
        trainer._build_training_records = mock.Mock(return_value=skipped_batch)
        trainer._attach_response_statistics = mock.Mock()
        trainer._build_generation_payload = mock.Mock(
            return_value={"reward_mean": 1.0, "generation_perf": {"generated_tokens_per_second": 1.0}}
        )
        logged_events = []
        trainer._log_event = lambda event, payload, **context: logged_events.append((event, payload, context))
        trainer._update_progress = mock.Mock()
        trainer.tree = SimpleNamespace(
            update_and_select=mock.Mock(return_value=([0], {"sampler/selected": 1.0})),
            select_batch=mock.Mock(return_value=([0], {"sampler/selected": 1.0})),
        )

        trainer._train_epoch(epoch=0, attempts_per_epoch=3)

        self.assertEqual(trainer.generation_attempt, 3)
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer._generate.call_count, 3)
        self.assertEqual(trainer.tree.update_and_select.call_count, 3)
        self.assertEqual([event for event, _, _ in logged_events], ["generation"] * 3)
        self.assertEqual(trainer._update_progress.call_count, 3)


if __name__ == "__main__":
    unittest.main()
