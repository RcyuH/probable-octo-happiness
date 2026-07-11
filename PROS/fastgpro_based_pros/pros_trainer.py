"""FastGRPO-based PROS trainer."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .pros_config import ProsConfig
from .pros_logging import (
    ProsEventLogger,
    RewardDebugAggregator,
    compute_generation_perf,
    compute_length_metrics,
    safe_div,
)
from .pros_loss import compute_gpg_advantages, compute_grpo_advantages, compute_policy_loss, gather_token_logps
from .pros_tree import ProsRolloutRecord, ProsTreeConfig, ProsTreeEngine

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal runtime environments.
    tqdm = None


@dataclass
class EncodedPrompt:
    item: int
    ancestor_item: int
    question: str
    answer: Optional[str]
    example: Dict[str, Any]
    task_id: str
    prompt_ids: List[int]
    partial_rollout: List[int]
    input_ids: List[int]

    @property
    def partial_rollout_len(self) -> int:
        return len(self.partial_rollout)


@dataclass
class TrainingRecord:
    item: int
    ancestor_item: int
    prompt_ids: List[int]
    partial_rollout: List[int]
    generated_ids: List[int]
    full_response_ids: List[int]
    full_input_ids: List[int]
    new_token_mask: List[int]
    reward: float
    decoded_completion: str
    task_id: str
    reward_example: Dict[str, Any]
    reward_detail: Dict[str, Any] = field(default_factory=dict)
    advantage: float = 0.0
    old_log_prob: Any = None
    ref_log_prob: Any = None
    entropies: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    @property
    def response_mask_for_tree(self) -> List[int]:
        return [1] * len(self.full_response_ids)


@dataclass
class RolloutBatch:
    """All verified rollouts plus the subset eligible for actor optimization."""

    all_records: List[TrainingRecord]
    actor_records: List[TrainingRecord]
    group_decisions: Dict[int, str]
    generated_groups: int = 0
    used_groups: int = 0
    skipped_correct_groups: int = 0
    skipped_incorrect_groups: int = 0
    empty_completions: int = 0


class ProsTrainer:
    """Orchestrates PROS training while reusing FastGRPO helpers."""

    def __init__(self, config: ProsConfig):
        self.cfg = config
        self.cfg.validate()
        self.repo_root = Path(__file__).resolve().parents[2]
        self.fastgrpo_root = self.repo_root / "FastGRPO"
        self._ensure_fastgrpo_importable()

        self.torch = self._import_torch()
        self._seed_everything(config.seed)
        self.device = self._resolve_device()

        self._load_fastgrpo_helpers()
        self._load_models()
        self.qas = self._load_training_examples()
        self.eval_qas = self._load_eval_examples()
        root_task_ids, task_weights = self._tree_task_metadata(self.qas)
        self.tree = ProsTreeEngine(
            len(self.qas),
            self._tree_config(),
            root_task_ids=root_task_ids,
            task_weights=task_weights,
        )
        self.next_items, self.last_sampler_metrics = self.tree.select_batch(config.batch_size, step_num=0)

        self.optimizer_target = self.torch.optim.AdamW(
            self._trainable_target_parameters(),
            lr=config.target_lr,
        )
        self.optimizer_draft = (
            self.torch.optim.AdamW(self.model.draft_model.parameters(), lr=config.draft_lr)
            if config.train_draft and config.generation_backend == "speculative"
            else None
        )
        self.target_accumulated_steps = 0
        self.draft_accumulated_steps = 0
        self.global_step = 0
        self.generation_attempt = 0
        self.used_groups = 0
        self.task_stats: Dict[str, Dict[str, float]] = {}
        self.generation_totals: Dict[str, float] = {
            "generation_time_sec": 0.0,
            "generated_completion_tokens": 0.0,
            "prefill_time_sec": 0.0,
            "speculative_emitted_tokens": 0.0,
            "speculative_accepted_draft_tokens": 0.0,
            "speculative_verified_draft_tokens": 0.0,
            "speculative_path_budget_tokens": 0.0,
            "speculative_verification_rounds": 0.0,
        }
        self.start_time = time.time()

        self._prepare_output_dirs()
        self.cfg.save()
        self.event_logger = ProsEventLogger(
            self.cfg.log_file,
            use_tensorboard=self.cfg.use_tensorboard,
            tensorboard_log_dir=self.cfg.tensorboard_log_dir,
        )
        self.reward_debug = RewardDebugAggregator(
            sample_limit=self.cfg.reward_debug_sample_count,
            char_limit=self.cfg.reward_debug_sample_chars,
        )
        self.progress = None
        self.event_logger.log(
            "config",
            {
                "config": self.cfg.to_dict(),
                "generation_backend": self.cfg.generation_backend,
                "lora_config": self._effective_lora_config(),
                "speculative_config": self._effective_speculative_config(),
                "tensorboard_enabled": self.event_logger.writer is not None,
                "tensorboard_log_dir_effective": str(self.event_logger.tensorboard_log_dir),
            },
            epoch=0,
            step=self.global_step,
            global_step=self.global_step,
            generation_attempt=self.generation_attempt,
            generation_backend=self.cfg.generation_backend,
        )

    def train(self) -> None:
        if self.cfg.dry_run:
            print(json.dumps({"dry_run": True, "config": self.cfg.to_dict()}, indent=2, sort_keys=True))
            self.event_logger.close()
            return

        attempts_per_epoch = max(1, math.ceil(len(self.qas) / self.cfg.batch_size))
        total_attempts = self.cfg.num_epochs * attempts_per_epoch
        if tqdm is not None:
            self.progress = tqdm(total=total_attempts, desc="PROS training", dynamic_ncols=True, unit="batch")
        else:
            print("tqdm is not installed; continuing without a progress bar.")

        try:
            for epoch in range(self.cfg.num_epochs):
                if self.cfg.max_train_steps and self.global_step >= self.cfg.max_train_steps:
                    break
                self._train_epoch(epoch, attempts_per_epoch)
            self._save_checkpoint(self.global_step)
        finally:
            if self.progress is not None:
                self.progress.close()
            self.event_logger.close()

    def _train_epoch(self, epoch: int, attempts_per_epoch: Optional[int] = None) -> None:
        attempts_per_epoch = attempts_per_epoch or max(1, math.ceil(len(self.qas) / self.cfg.batch_size))
        for _ in range(attempts_per_epoch):
            if self.cfg.max_train_steps and self.global_step >= self.cfg.max_train_steps:
                return
            self.generation_attempt += 1
            self.reward_debug.reset_batch()
            attempted_task_counts = self._task_counts_for_items(self.next_items)
            prompts = self._encode_selected_items(self.next_items)
            if not prompts:
                self.next_items, self.last_sampler_metrics = self.tree.select_batch(
                    self.cfg.batch_size,
                    self.generation_attempt,
                )
                self._log_empty_generation(
                    epoch,
                    reason="no_valid_prompts",
                    task_prompt_counts=attempted_task_counts,
                )
                self._update_progress(
                    {
                        "step": self.global_step,
                        "skip": sum(attempted_task_counts.values()),
                        "reward": 0.0,
                        "tok/s": 0.0,
                        "tasks": self._format_task_composition(attempted_task_counts),
                    }
                )
                continue

            generation_start = time.time()
            outputs = self._generate(prompts)
            generation_time = time.time() - generation_start

            reward_start = time.time()
            rollout_batch = self._build_training_records(prompts, outputs)
            reward_time = time.time() - reward_start

            if (
                self.cfg.train_draft
                and self.cfg.generation_backend == "speculative"
                and self.optimizer_draft is not None
            ):
                draft_start = time.time()
                draft_metrics = self._train_draft_model(outputs, prompts)
                draft_metrics["draft/train_time_sec"] = time.time() - draft_start
            else:
                draft_metrics = {}

            stats_start = time.time()
            if rollout_batch.all_records:
                self._attach_response_statistics(rollout_batch.all_records)
            response_stats_time = time.time() - stats_start

            tree_records = [
                ProsRolloutRecord(
                    item=record.item,
                    response_ids=record.full_response_ids,
                    reward=record.reward,
                    response_mask=record.response_mask_for_tree,
                    partial_rollout_len=len(record.partial_rollout),
                    entropies=record.entropies,
                    values=record.values or None,
                )
                for record in rollout_batch.all_records
            ]
            tree_start = time.time()
            if tree_records:
                self.next_items, tree_metrics = self.tree.update_and_select(
                    tree_records,
                    step_num=self.generation_attempt,
                    batch_size=self.cfg.batch_size,
                )
            else:
                self.next_items, tree_metrics = self.tree.select_batch(
                    self.cfg.batch_size,
                    step_num=self.generation_attempt,
                )
            self.last_sampler_metrics = tree_metrics
            tree_time = time.time() - tree_start

            generation_payload = self._build_generation_payload(
                epoch=epoch,
                prompts=prompts,
                outputs=outputs,
                rollout_batch=rollout_batch,
                generation_time=generation_time,
                reward_time=reward_time,
                response_stats_time=response_stats_time,
                tree_time=tree_time,
                tree_metrics=tree_metrics,
                draft_metrics=draft_metrics,
            )
            self._log_event(
                "generation",
                generation_payload,
                epoch=epoch + 1,
                step=self.global_step,
                global_step=self.global_step,
                generation_attempt=self.generation_attempt,
                generation_backend=self.cfg.generation_backend,
            )

            train_metrics: Dict[str, Any] = {}
            if rollout_batch.actor_records:
                self._attach_advantages(rollout_batch.actor_records)
                actor_start = time.time()
                train_metrics = self._train_actor(rollout_batch.actor_records)
                train_metrics["actor/train_time_sec"] = time.time() - actor_start
                self.global_step += 1
                train_payload = {
                    "used_groups": self.used_groups,
                    "actor_records": len(rollout_batch.actor_records),
                    "elapsed_min": round((time.time() - self.start_time) / 60.0, 4),
                    "generation_perf": self._cumulative_generation_perf(),
                    "task_metrics": self._summarize_task_stats(),
                    "reward_debug": self.reward_debug.snapshot(),
                    "lora_config": self._effective_lora_config(),
                    "speculative_config": self._effective_speculative_config(),
                    **tree_metrics,
                    **train_metrics,
                    **draft_metrics,
                }
                self._log_event(
                    "train",
                    train_payload,
                    epoch=epoch + 1,
                    step=self.global_step,
                    global_step=self.global_step,
                    generation_attempt=self.generation_attempt,
                    generation_backend=self.cfg.generation_backend,
                )

                if self.cfg.eval_freq > 0 and self.eval_qas and self.global_step % self.cfg.eval_freq == 0:
                    eval_metrics = self._evaluate()
                    self._log_event(
                        "eval",
                        eval_metrics,
                        epoch=epoch + 1,
                        step=self.global_step,
                        global_step=self.global_step,
                        generation_attempt=self.generation_attempt,
                        generation_backend=self.cfg.generation_backend,
                    )
                if self.cfg.save_freq > 0 and self.global_step % self.cfg.save_freq == 0:
                    self._save_checkpoint(self.global_step)

            self._update_progress(
                {
                    "step": self.global_step,
                    "reward": generation_payload.get("reward_mean", 0.0),
                    "tok/s": generation_payload["generation_perf"].get("generated_tokens_per_second", 0.0),
                    "skip": rollout_batch.skipped_correct_groups + rollout_batch.skipped_incorrect_groups,
                    "tasks": self._format_task_composition(
                        generation_payload.get("task_prompt_counts", {})
                    ),
                }
            )

    def _effective_lora_config(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.cfg.use_lora),
            "r": self.cfg.lora_r,
            "lora_alpha": self.cfg.lora_alpha,
            "lora_dropout": self.cfg.lora_dropout,
            "target_modules": list(self.cfg.lora_target_modules),
            "bias": self.cfg.lora_bias,
            "load_lora_path": self.cfg.load_lora_path,
        }

    def _trainable_target_parameters(self) -> List[Any]:
        """Return exactly the parameters the target optimizer may update."""

        return [parameter for parameter in self.model.target_model.parameters() if parameter.requires_grad]

    def _effective_speculative_config(self) -> Dict[str, Any]:
        return {
            "verification_capacity": self.cfg.verification_capacity,
            "max_draft_token_length": self.cfg.max_draft_token_length,
            "min_draft_token_length": self.cfg.min_draft_token_length,
            "max_draft_k": self.cfg.max_draft_k,
            "max_verification_num": self.cfg.max_verification_num,
            "draft_token_length_c": self.cfg.draft_token_length_c,
        }

    def _tree_task_metadata(
        self,
        examples: Sequence[Dict[str, Any]],
    ) -> Tuple[List[str], Optional[Dict[str, float]]]:
        root_task_ids = [str(example.get("task_id", "default")) for example in examples]
        if not self.has_explicit_task_weights(examples):
            return root_task_ids, None

        task_weights: Dict[str, float] = {}
        for example, task_id in zip(examples, root_task_ids):
            weight = float(example.get("task_weight", 1.0))
            previous = task_weights.setdefault(task_id, weight)
            if not math.isclose(previous, weight, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Conflicting task weights for {task_id!r}: {previous} and {weight}")
        return root_task_ids, task_weights

    def _update_progress(self, values: Dict[str, Any]) -> None:
        if self.progress is None:
            return
        self.progress.set_postfix(values)
        self.progress.update(1)

    @staticmethod
    def _format_task_composition(task_counts: Dict[str, Any], max_chars: int = 80) -> str:
        summary = ",".join(
            f"{task_id}:{int(count)}" for task_id, count in sorted(task_counts.items())
        )
        if len(summary) <= max_chars:
            return summary or "-"
        return summary[: max(max_chars - 3, 0)] + "..."

    def _task_counts_for_items(self, items: Sequence[int]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        get_task_id = getattr(self.tree, "get_task_id", None)
        if not callable(get_task_id):
            return counts
        for item in items:
            task_id = str(get_task_id(int(item)))
            counts[task_id] = counts.get(task_id, 0) + 1
        return counts

    def _log_event(self, event: str, payload: Dict[str, Any], **context: Any) -> Dict[str, Any]:
        record = self.event_logger.log(event, payload, **context)
        if self.cfg.log_freq > 0 and self.generation_attempt % self.cfg.log_freq == 0:
            perf = record.get("generation_perf") if isinstance(record.get("generation_perf"), dict) else {}
            summary = {
                "event": event,
                "epoch": record.get("epoch"),
                "step": record.get("step"),
                "generation_attempt": record.get("generation_attempt"),
                "reward_mean": record.get("reward_mean", record.get("eval/reward_mean")),
                "generated_tokens_per_second": perf.get("generated_tokens_per_second"),
                "actor_loss": record.get("actor/loss"),
            }
            print(json.dumps({key: value for key, value in summary.items() if value is not None}, sort_keys=True))
        return record

    def _log_empty_generation(
        self,
        epoch: int,
        *,
        reason: str,
        task_prompt_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        outputs = {
            "total_time_cost": 0.0,
            "target_time_cost": 0.0,
            "draft_time_cost": 0.0,
            "check_time_cost": 0.0,
        }
        payload = {
            "reason": reason,
            "batch_prompt_count": 0,
            "batch_completion_count": 0,
            "generated_group_count": 0,
            "batch_used_group_count": 0,
            "batch_ignore_due_correct": 0,
            "batch_ignore_due_incorrect": 0,
            "empty_completion_count": 0,
            "actor_eligible_completion_count": 0,
            "task_prompt_counts": dict(task_prompt_counts or {}),
            "task_completion_counts": {},
            "generation_perf": compute_generation_perf(
                outputs,
                [],
                generation_time_sec=0.0,
                generation_backend=self.cfg.generation_backend,
            ),
            "task_metrics": self._summarize_task_stats(),
            "phase_timing": {
                "generation_time_sec": 0.0,
                "reward_time_sec": 0.0,
                "draft_train_time_sec": 0.0,
                "response_statistics_time_sec": 0.0,
                "tree_update_time_sec": 0.0,
            },
            "lora_config": self._effective_lora_config(),
            "speculative_config": self._effective_speculative_config(),
            **compute_length_metrics([], [], []),
            **self.reward_debug.as_payload(),
            **self.last_sampler_metrics,
        }
        self._log_event(
            "generation",
            payload,
            epoch=epoch + 1,
            step=self.global_step,
            global_step=self.global_step,
            generation_attempt=self.generation_attempt,
            generation_backend=self.cfg.generation_backend,
        )

    def _build_generation_payload(
        self,
        *,
        epoch: int,
        prompts: Sequence[EncodedPrompt],
        outputs: Dict[str, Any],
        rollout_batch: RolloutBatch,
        generation_time: float,
        reward_time: float,
        response_stats_time: float,
        tree_time: float,
        tree_metrics: Dict[str, Any],
        draft_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        suffix_lengths = [len(record.generated_ids) for record in rollout_batch.all_records]
        partial_lengths = [len(record.partial_rollout) for record in rollout_batch.all_records]
        full_lengths = [len(record.full_response_ids) for record in rollout_batch.all_records]
        rewards = [record.reward for record in rollout_batch.all_records]
        generation_perf = compute_generation_perf(
            outputs,
            suffix_lengths,
            generation_time_sec=generation_time,
            generation_backend=self.cfg.generation_backend,
        )
        self._accumulate_generation_perf(generation_perf)
        task_completion_counts: Dict[str, int] = {}
        for record in rollout_batch.all_records:
            task_completion_counts[record.task_id] = task_completion_counts.get(record.task_id, 0) + 1
        task_prompt_counts: Dict[str, int] = {}
        for prompt in prompts:
            task_prompt_counts[prompt.task_id] = task_prompt_counts.get(prompt.task_id, 0) + 1

        return {
            "batch_prompt_count": len(prompts),
            "batch_completion_count": len(rollout_batch.all_records),
            "actor_eligible_completion_count": len(rollout_batch.actor_records),
            "generated_group_count": rollout_batch.generated_groups,
            "batch_used_group_count": rollout_batch.used_groups,
            "batch_ignore_due_correct": rollout_batch.skipped_correct_groups,
            "batch_ignore_due_incorrect": rollout_batch.skipped_incorrect_groups,
            "empty_completion_count": rollout_batch.empty_completions,
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "task_prompt_counts": task_prompt_counts,
            "task_completion_counts": task_completion_counts,
            "group_decisions": dict(rollout_batch.group_decisions),
            "generation_perf": generation_perf,
            "phase_timing": {
                "generation_time_sec": generation_time,
                "reward_time_sec": reward_time,
                "draft_train_time_sec": float(draft_metrics.get("draft/train_time_sec", 0.0)),
                "response_statistics_time_sec": response_stats_time,
                "tree_update_time_sec": tree_time,
            },
            "task_metrics": self._summarize_task_stats(),
            "lora_config": self._effective_lora_config(),
            "speculative_config": self._effective_speculative_config(),
            **compute_length_metrics(suffix_lengths, partial_lengths, full_lengths),
            **self.reward_debug.as_payload(),
            **tree_metrics,
            **draft_metrics,
        }

    def _accumulate_generation_perf(self, perf: Dict[str, Any]) -> None:
        for key in (
            "generation_time_sec",
            "generated_completion_tokens",
            "target_time_sec",
            "draft_time_sec",
            "check_time_sec",
            "prefill_time_sec",
            "speculative_emitted_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_verified_draft_tokens",
            "speculative_path_budget_tokens",
            "speculative_verification_rounds",
        ):
            self.generation_totals[key] = self.generation_totals.get(key, 0.0) + float(perf.get(key, 0.0) or 0.0)

    def _cumulative_generation_perf(self) -> Dict[str, float]:
        totals = dict(self.generation_totals)
        generation_time = totals.get("generation_time_sec", 0.0)
        rounds = totals.get("speculative_verification_rounds", 0.0)
        accepted = totals.get("speculative_accepted_draft_tokens", 0.0)
        emitted = totals.get("speculative_emitted_tokens", 0.0)
        verified = totals.get("speculative_verified_draft_tokens", 0.0)
        path_budget = totals.get("speculative_path_budget_tokens", 0.0)
        return {
            **totals,
            "generated_tokens_per_second": safe_div(totals.get("generated_completion_tokens", 0.0), generation_time),
            "target_time_ratio": safe_div(totals.get("target_time_sec", 0.0), generation_time),
            "draft_time_ratio": safe_div(totals.get("draft_time_sec", 0.0), generation_time),
            "check_time_ratio": safe_div(totals.get("check_time_sec", 0.0), generation_time),
            "prefill_time_ratio": safe_div(totals.get("prefill_time_sec", 0.0), generation_time),
            "speculative_avg_emitted_tokens_per_round": safe_div(emitted, rounds),
            "speculative_avg_accepted_draft_tokens_per_round": safe_div(accepted, rounds),
            "speculative_path_acceptance_rate": safe_div(accepted, path_budget),
            "speculative_tree_acceptance_rate": safe_div(accepted, verified),
            "speculative_verified_draft_tokens_per_round": safe_div(verified, rounds),
        }

    def _ensure_fastgrpo_importable(self) -> None:
        fastgrpo_path = str(self.fastgrpo_root)
        if fastgrpo_path not in sys.path:
            sys.path.insert(0, fastgrpo_path)

    def _import_torch(self):
        try:
            import torch

            return torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install FastGRPO requirements before training PROS: torch is missing.") from exc

    def _load_fastgrpo_helpers(self) -> None:
        from helper.get_QAs import get_test_QAs, get_train_QAs
        from helper.modeling_draft import Model
        from helper.multitask import (
            compute_multitask_reward_debug,
            has_explicit_task_weights,
            load_multitask_QAs,
            normalize_single_task_QAs,
            render_messages,
        )
        from helper.specualtive_generate import speculative_generate

        self.Model = Model
        self.get_train_QAs = get_train_QAs
        self.get_test_QAs = get_test_QAs
        self.compute_multitask_reward_debug = compute_multitask_reward_debug
        self.has_explicit_task_weights = has_explicit_task_weights
        self.load_multitask_QAs = load_multitask_QAs
        self.normalize_single_task_QAs = normalize_single_task_QAs
        self.render_messages = render_messages
        self.speculative_generate = speculative_generate

    def _load_training_examples(self) -> List[Dict[str, Any]]:
        if self.cfg.task_config:
            samples_per_epoch = self.cfg.task_samples_per_epoch or None
            return self.load_multitask_QAs(
                self.cfg.task_config,
                split=self.cfg.task_split,
                samples_per_epoch=samples_per_epoch,
                seed=self.cfg.seed,
            )
        return self.normalize_single_task_QAs(
            self.get_train_QAs(self.cfg.train_option),
            task_id=self.cfg.train_option,
            prompt_type="math",
            reward_type="math_latex",
        )

    def _load_eval_examples(self) -> List[Dict[str, Any]]:
        if self.cfg.eval_task_config:
            return self.load_multitask_QAs(
                self.cfg.eval_task_config,
                split=self.cfg.eval_task_split,
                samples_per_epoch=self.cfg.eval_samples or None,
                seed=self.cfg.seed,
            )
        if self.cfg.eval_option:
            return self.normalize_single_task_QAs(
                self.get_test_QAs(self.cfg.eval_option),
                task_id=self.cfg.eval_option,
                prompt_type="math",
                reward_type="math_latex",
            )
        return []

    def _load_models(self) -> None:
        if not self.cfg.model_dir:
            raise ValueError("--model-dir is required for training")

        from copy import deepcopy

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_config = AutoConfig.from_pretrained(self.cfg.model_dir)
        target_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_dir,
            torch_dtype="auto",
            config=model_config,
        ).to(self.device)
        target_model.eval()

        draft_config = deepcopy(model_config)
        draft_config.rope_scaling = None
        draft_config.num_hidden_layers = 1
        self.model = self.Model(draft_config, target_model=target_model).to(self.device)
        if self.cfg.adapter_path:
            self.model.load_model(self.cfg.adapter_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_dir, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if model_config.model_type == "llama" and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = "<|end_of_text|>"
            self.tokenizer.pad_token_id = 128001

        for param in self.model.draft_model.parameters():
            param.requires_grad = bool(self.cfg.train_draft)
        if self.cfg.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            for param in self.model.target_model.parameters():
                param.requires_grad = False
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=list(self.cfg.lora_target_modules),
                bias=self.cfg.lora_bias,
            )
            self.model.target_model = get_peft_model(self.model.target_model, lora_config)
            if self.cfg.load_lora_path:
                self.model.target_model.load_adapter(self.cfg.load_lora_path, adapter_name="default")
            if hasattr(self.model.target_model, "print_trainable_parameters"):
                self.model.target_model.print_trainable_parameters()
        else:
            for param in self.model.target_model.parameters():
                param.requires_grad = True

    def _resolve_device(self):
        if self.torch.cuda.is_available():
            return self.torch.device("cuda")
        if self.cfg.allow_cpu:
            return self.torch.device("cpu")
        raise RuntimeError(
            "FastGRPO speculative generation is CUDA-oriented. Re-run with CUDA or use --allow-cpu only for debugging."
        )

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    def _tree_config(self) -> ProsTreeConfig:
        return ProsTreeConfig(
            selector=self.cfg.tree_selector,
            sampler=self.cfg.tree_sampler,
            mu0=self.cfg.tree_mu0,
            tau0=self.cfg.tree_tau0,
            sigma0=self.cfg.tree_sigma0,
            delta=self.cfg.tree_delta,
            gamma=self.cfg.tree_gamma,
            gibbs_sweeps=self.cfg.tree_gibbs_sweeps,
            min_window_tokens=self.cfg.tree_min_window_tokens,
            score_threshold=self.cfg.tree_score_threshold,
            allow_fallback_fill=self.cfg.tree_allow_fallback_fill,
            random_seed=self.cfg.seed,
        )

    def _prepare_output_dirs(self) -> None:
        for path in [
            self.cfg.output_dir,
            os.path.dirname(self.cfg.log_file) or ".",
            self.cfg.saved_model_dir,
            self.cfg.saved_draft_model_dir,
            self.cfg.saved_statistics_dir,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _encode_selected_items(self, items: Sequence[int]) -> List[EncodedPrompt]:
        prompts: List[EncodedPrompt] = []
        for item in items:
            node = self.tree.get_node(int(item))
            ancestor = self.tree.get_original_ancestor_item(int(item))
            qa = self.qas[ancestor]
            prompt_ids = self._encode_base_prompt(qa)
            partial = node.partial_rollout or []
            input_ids = prompt_ids + partial
            if len(input_ids) >= self.cfg.max_length:
                continue
            prompts.append(
                EncodedPrompt(
                    item=int(item),
                    ancestor_item=ancestor,
                    question=qa.get("question") or qa.get("prompt") or qa.get("instruction") or "",
                    answer=qa.get("answer"),
                    example=qa,
                    task_id=qa.get("task_id", "default"),
                    prompt_ids=prompt_ids,
                    partial_rollout=partial,
                    input_ids=input_ids,
                )
            )
        return prompts

    def _encode_base_prompt(self, example: Dict[str, Any]) -> List[int]:
        text = self.tokenizer.apply_chat_template(
            self.render_messages(example),
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _pad_left(self, rows: Sequence[List[int]]) -> Tuple[Any, Any]:
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        max_len = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad_len = max_len - len(row)
            input_ids.append([pad_id] * pad_len + row)
            attention_mask.append([0] * pad_len + [1] * len(row))
        return (
            self.torch.tensor(input_ids, device=self.device, dtype=self.torch.long),
            self.torch.tensor(attention_mask, device=self.device, dtype=self.torch.long),
        )

    def _pad_right(self, rows: Sequence[List[int]], pad_value: int = 0, dtype: Any = None) -> Any:
        max_len = max(len(row) for row in rows)
        padded = [row + [pad_value] * (max_len - len(row)) for row in rows]
        return self.torch.tensor(padded, device=self.device, dtype=dtype or self.torch.long)

    def _pad_right_with_attention(self, rows: Sequence[List[int]], pad_value: int = 0) -> Tuple[Any, Any]:
        max_len = max(len(row) for row in rows)
        padded = []
        attention = []
        for row in rows:
            pad_len = max_len - len(row)
            padded.append(row + [pad_value] * pad_len)
            attention.append([1] * len(row) + [0] * pad_len)
        return (
            self.torch.tensor(padded, device=self.device, dtype=self.torch.long),
            self.torch.tensor(attention, device=self.device, dtype=self.torch.long),
        )

    def _generate(self, prompts: Sequence[EncodedPrompt]) -> Dict[str, Any]:
        self.model.target_model.eval()
        if hasattr(self.model, "draft_model"):
            self.model.draft_model.eval()
        input_ids, attention_mask = self._pad_left([prompt.input_ids for prompt in prompts])
        statistical_time = bool(self.torch.cuda.is_available())
        top_k = self.cfg.top_k if self.cfg.top_k > 0 else None
        if self.cfg.generation_backend == "target":
            return self._target_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                repeated_generate_nums=self.cfg.repeated_generate_nums,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=top_k,
                statistical_time=statistical_time,
                max_length=self.cfg.max_length,
            )
        # Draft-training states are consumed by modules whose weights require
        # gradients.  `inference_mode` tensors cannot always be saved for that
        # backward pass, whereas `no_grad` avoids graph construction while
        # returning ordinary tensors.  Keep the stronger mode when no draft
        # update will consume the returned states.
        generation_context = self.torch.no_grad if self.cfg.train_draft else self.torch.inference_mode
        with generation_context():
            return self.speculative_generate(
                model=self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=self.tokenizer,
                do_sample=True,
                max_length=self.cfg.max_length,
                repeated_generate_nums=self.cfg.repeated_generate_nums,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=top_k,
                verification_capacity=self.cfg.verification_capacity,
                max_draft_token_length=self.cfg.max_draft_token_length,
                min_draft_token_length=self.cfg.min_draft_token_length,
                max_draft_k=self.cfg.max_draft_k,
                max_verification_num=self.cfg.max_verification_num,
                draft_token_length_c=self.cfg.draft_token_length_c,
                return_all_draft_input=self.cfg.train_draft,
                statistical_time=statistical_time,
            )

    def _target_generate(
        self,
        input_ids,
        attention_mask,
        *,
        do_sample: bool,
        repeated_generate_nums: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        statistical_time: bool,
        max_length: int,
    ) -> Dict[str, Any]:
        start_time = time.time()
        repeated_nums = repeated_generate_nums or 1
        prompt_length = input_ids.shape[-1]
        max_new_tokens = max(max_length - prompt_length, 1)
        expanded_input_ids = input_ids.to(self.device).repeat_interleave(repeated_nums, dim=0)
        expanded_attention_mask = attention_mask.to(self.device).repeat_interleave(repeated_nums, dim=0)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        generation_kwargs = {
            "input_ids": expanded_input_ids,
            "attention_mask": expanded_attention_mask,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            if top_k is not None:
                generation_kwargs["top_k"] = top_k

        target_time_start = time.time()
        with self.torch.inference_mode():
            output_ids = self.model.target_model.generate(**generation_kwargs)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        target_time_cost = time.time() - target_time_start

        generated_token_ids = []
        for sequence in output_ids:
            completion = sequence[prompt_length:].detach().cpu().tolist()
            trimmed = []
            for token in completion:
                if self.tokenizer.eos_token_id is not None and token == self.tokenizer.eos_token_id:
                    trimmed.append(token)
                    break
                if pad_token_id is not None and token == pad_token_id:
                    continue
                trimmed.append(token)
            generated_token_ids.append(trimmed)

        total_decoded_token_num = sum(len(item) for item in generated_token_ids)
        max_sequence_length = max((len(item) for item in generated_token_ids), default=0)
        total_time_cost = time.time() - start_time
        return {
            "generated_token_ids": generated_token_ids,
            "max_sequence_length": max_sequence_length,
            "total_acc_length": total_decoded_token_num,
            "total_acc": 1.0,
            "total_decoded_token_num": max(total_decoded_token_num, 1),
            "speculative_emitted_tokens": 0,
            "speculative_accepted_draft_tokens": 0,
            "speculative_verified_draft_tokens": 0,
            "speculative_path_budget_tokens": 0,
            "speculative_verification_rounds": 0,
            "total_time_cost": total_time_cost,
            "target_time_cost": target_time_cost,
            "draft_time_cost": 0.0,
            "check_time_cost": 0.0,
            "prefill_time_cost": 0.0,
            "post_time_cost": total_time_cost - target_time_cost,
            "all_draft_input_states": None,
            "all_draft_input_ids": None,
        }

    def _get_task_stat(self, task_id: str) -> Dict[str, float]:
        if task_id not in self.task_stats:
            self.task_stats[task_id] = {
                "used_items": 0.0,
                "reward_sum": 0.0,
                "reward_count": 0.0,
                "all_reward_sum": 0.0,
                "all_reward_count": 0.0,
                "generated_length_sum": 0.0,
                "generated_completion_count": 0.0,
                "empty_completion_count": 0.0,
                "ignore_due_correct": 0.0,
                "ignore_due_incorrect": 0.0,
            }
        return self.task_stats[task_id]

    def _summarize_task_stats(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for task_id, stats in self.task_stats.items():
            reward_count = stats["reward_count"]
            all_reward_count = stats["all_reward_count"]
            completion_count = stats["generated_completion_count"]
            summary[task_id] = {
                "used_items": int(stats["used_items"]),
                "mean_reward": round(stats["reward_sum"] / reward_count, 4) if reward_count else 0.0,
                "mean_reward_all_completions": (
                    round(stats["all_reward_sum"] / all_reward_count, 4) if all_reward_count else 0.0
                ),
                "mean_length": round(stats["generated_length_sum"] / completion_count, 3) if completion_count else 0.0,
                "generated_completions": int(completion_count),
                "empty_completions": int(stats["empty_completion_count"]),
                "ignore_due_correct": int(stats["ignore_due_correct"]),
                "ignore_due_incorrect": int(stats["ignore_due_incorrect"]),
            }
        return summary

    def _evaluate(self) -> Dict[str, Any]:
        eval_examples = self.eval_qas[: self.cfg.eval_samples]
        if not eval_examples:
            return {}

        self.model.target_model.eval()
        if hasattr(self.model, "draft_model"):
            self.model.draft_model.eval()
        eval_reward_debug = RewardDebugAggregator(
            sample_limit=self.cfg.reward_debug_sample_count,
            char_limit=self.cfg.reward_debug_sample_chars,
        )
        prompts = []
        for i, example in enumerate(eval_examples):
            prompt_ids = self._encode_base_prompt(example)
            prompts.append(
                EncodedPrompt(
                    item=i,
                    ancestor_item=i,
                    question=example.get("question") or example.get("prompt") or example.get("instruction") or "",
                    answer=example.get("answer"),
                    example=example,
                    task_id=example.get("task_id", "default"),
                    prompt_ids=prompt_ids,
                    partial_rollout=[],
                    input_ids=prompt_ids,
                )
            )
        rewards: List[float] = []
        accuracies: List[float] = []
        task_rewards: Dict[str, List[float]] = {}
        for start in range(0, len(prompts), self.cfg.batch_size):
            batch_prompts = prompts[start : start + self.cfg.batch_size]
            input_ids, attention_mask = self._pad_left([prompt.input_ids for prompt in batch_prompts])
            with self.torch.inference_mode():
                top_k = self.cfg.top_k if self.cfg.top_k > 0 else None
                if self.cfg.generation_backend == "target":
                    outputs = self._target_generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        max_length=self.cfg.max_length,
                        repeated_generate_nums=1,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        top_k=top_k,
                        statistical_time=bool(self.torch.cuda.is_available()),
                    )
                else:
                    outputs = self.speculative_generate(
                        model=self.model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        tokenizer=self.tokenizer,
                        do_sample=True,
                        max_length=self.cfg.max_length,
                        repeated_generate_nums=1,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        top_k=top_k,
                        verification_capacity=self.cfg.verification_capacity,
                        max_draft_token_length=self.cfg.max_draft_token_length,
                        min_draft_token_length=self.cfg.min_draft_token_length,
                        max_draft_k=self.cfg.max_draft_k,
                        max_verification_num=self.cfg.max_verification_num,
                        draft_token_length_c=self.cfg.draft_token_length_c,
                        return_all_draft_input=False,
                        statistical_time=bool(self.torch.cuda.is_available()),
                    )
            expected_outputs = len(batch_prompts)
            if len(outputs.get("generated_token_ids", [])) != expected_outputs:
                raise ValueError(
                    "Generation output count mismatch during evaluation: "
                    f"expected {expected_outputs}, got {len(outputs.get('generated_token_ids', []))}"
                )
            for prompt, generated_ids in zip(batch_prompts, outputs["generated_token_ids"]):
                completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                reward_detail = self.compute_multitask_reward_debug(completion, prompt.example)
                reward = float(reward_detail["reward"])
                eval_reward_debug.record_completion(
                    reward_detail,
                    task_id=prompt.task_id,
                    prompt=self.render_messages(prompt.example),
                    completion=completion,
                )
                rewards.append(reward)
                accuracies.append(float(bool(reward_detail.get("passed", reward >= 1.0))))
                task_rewards.setdefault(prompt.task_id, []).append(reward)
        return {
            "eval/reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "eval/accuracy_mean": float(np.mean(accuracies)) if accuracies else 0.0,
            "eval/samples": float(len(rewards)),
            "eval/task_reward_mean": {
                task_id: float(np.mean(values)) if values else 0.0 for task_id, values in task_rewards.items()
            },
            **eval_reward_debug.as_payload(),
        }

    def _build_training_records(self, prompts: Sequence[EncodedPrompt], outputs: Dict[str, Any]) -> RolloutBatch:
        generated = list(outputs.get("generated_token_ids", []))
        prompt_records: Dict[int, List[TrainingRecord]] = {}
        repeat_count = self.cfg.repeated_generate_nums or 1
        expected_outputs = len(prompts) * repeat_count
        if len(generated) != expected_outputs:
            raise ValueError(
                "Generation output count mismatch: "
                f"expected {expected_outputs} for {len(prompts)} prompts x {repeat_count} repeats, "
                f"got {len(generated)}"
            )

        empty_completions = 0

        for idx, generated_ids in enumerate(generated):
            prompt_idx = idx // repeat_count
            prompt = prompts[prompt_idx]
            generated_ids = list(generated_ids)
            if not generated_ids:
                empty_completions += 1
            full_response_ids = prompt.partial_rollout + generated_ids
            decoded_completion = self.tokenizer.decode(full_response_ids, skip_special_tokens=True)
            reward_detail = self.compute_multitask_reward_debug(decoded_completion, prompt.example)
            if not isinstance(reward_detail, dict) or "reward" not in reward_detail:
                raise TypeError("compute_multitask_reward_debug must return a dict containing 'reward'")
            reward = float(reward_detail["reward"])
            self.reward_debug.record_completion(
                reward_detail,
                task_id=prompt.task_id,
                prompt=self.render_messages(prompt.example),
                completion=decoded_completion,
                repeat_index=idx % repeat_count,
            )
            full_input_ids = prompt.prompt_ids + full_response_ids
            new_token_mask = [0] * (len(prompt.prompt_ids) + len(prompt.partial_rollout)) + [1] * len(generated_ids)
            if len(full_input_ids) != len(new_token_mask):
                raise AssertionError("full_input_ids and new_token_mask length mismatch")
            prompt_records.setdefault(prompt.item, []).append(
                TrainingRecord(
                    item=prompt.item,
                    ancestor_item=prompt.ancestor_item,
                    prompt_ids=prompt.prompt_ids,
                    partial_rollout=prompt.partial_rollout,
                    generated_ids=generated_ids,
                    full_response_ids=full_response_ids,
                    full_input_ids=full_input_ids,
                    new_token_mask=new_token_mask,
                    reward=reward,
                    decoded_completion=decoded_completion,
                    task_id=prompt.task_id,
                    reward_example=prompt.example,
                    reward_detail=dict(reward_detail),
                )
            )

        all_records: List[TrainingRecord] = []
        actor_records: List[TrainingRecord] = []
        group_decisions: Dict[int, str] = {}
        skipped_correct_groups = 0
        skipped_incorrect_groups = 0
        used_groups = 0
        for item, group in prompt_records.items():
            all_records.extend(group)
            rewards = [record.reward for record in group]
            eligible_group = [record for record in group if record.generated_ids]
            eligible_rewards = [record.reward for record in eligible_group]
            task_id = group[0].task_id if group else "default"
            task_stat = self._get_task_stat(task_id)
            task_stat["generated_length_sum"] += float(sum(len(record.generated_ids) for record in group))
            task_stat["generated_completion_count"] += float(len(group))
            task_stat["empty_completion_count"] += float(sum(not record.generated_ids for record in group))
            task_stat["all_reward_sum"] += float(sum(rewards))
            task_stat["all_reward_count"] += float(len(rewards))
            details = [record.reward_detail for record in group]
            if not eligible_group:
                if rewards and all(reward >= self.cfg.tree_score_threshold for reward in rewards):
                    decision = "ignore_due_correct"
                    task_stat["ignore_due_correct"] += 1.0
                    skipped_correct_groups += 1
                else:
                    decision = "ignore_due_incorrect"
                    task_stat["ignore_due_incorrect"] += 1.0
                    skipped_incorrect_groups += 1
                group_decisions[item] = decision
                self.reward_debug.record_group_decision(decision, details, task_id=task_id)
                continue
            if (
                self.cfg.drop_zero_std_groups
                and len(eligible_group) > 1
                and float(np.std(eligible_rewards)) == 0.0
            ):
                if eligible_rewards[0] >= self.cfg.tree_score_threshold:
                    decision = "ignore_due_correct"
                    task_stat["ignore_due_correct"] += 1.0
                    skipped_correct_groups += 1
                else:
                    decision = "ignore_due_incorrect"
                    task_stat["ignore_due_incorrect"] += 1.0
                    skipped_incorrect_groups += 1
                group_decisions[item] = decision
                self.reward_debug.record_group_decision(decision, details, task_id=task_id)
                continue
            decision = "used"
            group_decisions[item] = decision
            actor_records.extend(eligible_group)
            task_stat["used_items"] += 1.0
            task_stat["reward_sum"] += float(sum(eligible_rewards))
            task_stat["reward_count"] += float(len(eligible_rewards))
            self.used_groups += 1
            used_groups += 1
            self.reward_debug.record_group_decision(decision, details, task_id=task_id)
        return RolloutBatch(
            all_records=all_records,
            actor_records=actor_records,
            group_decisions=group_decisions,
            generated_groups=len(prompt_records),
            used_groups=used_groups,
            skipped_correct_groups=skipped_correct_groups,
            skipped_incorrect_groups=skipped_incorrect_groups,
            empty_completions=empty_completions,
        )

    def _attach_advantages(self, records: Sequence[TrainingRecord]) -> None:
        rewards = self.torch.tensor([record.reward for record in records], device=self.device, dtype=self.torch.float32)
        one_token_mask = self.torch.ones((len(records), 1), device=self.device, dtype=self.torch.float32)
        group_ids = [record.item for record in records]
        if self.cfg.advantage_estimator == "gpg":
            advantages = compute_gpg_advantages(rewards, one_token_mask, group_ids)
        elif self.cfg.advantage_estimator == "grpo":
            advantages = compute_grpo_advantages(rewards, one_token_mask, group_ids)
        else:
            raise ValueError(f"Unknown advantage estimator: {self.cfg.advantage_estimator}")
        advantages = advantages.detach().view(-1).cpu().tolist()
        for record, advantage in zip(records, advantages):
            record.advantage = float(advantage)

    def _attach_response_statistics(self, records: Sequence[TrainingRecord]) -> None:
        self.model.target_model.eval()
        with self.torch.no_grad():
            for batch in self._iter_packed_records(records):
                input_ids, attention_mask = self._pad_right_with_attention(
                    [record.full_input_ids for record in batch],
                    self.tokenizer.pad_token_id or 0,
                )
                outputs = self.model.target_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :].float()
                probs = logits.softmax(dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                log_prob = gather_token_logps(outputs.logits, input_ids)
                for row, record in enumerate(batch):
                    start = len(record.prompt_ids) - 1
                    end = start + len(record.full_response_ids)
                    record.entropies = entropy[row, start:end].detach().float().cpu().tolist()
                    record.values = log_prob[row, start:end].detach().float().cpu().tolist()

    def _train_actor(self, records: Sequence[TrainingRecord]) -> Dict[str, float]:
        self.model.target_model.train()
        all_stats: List[Dict[str, float]] = []
        for iteration in range(self.cfg.grpo_iteration_num):
            for batch in self._iter_packed_records(records):
                stats = self._train_actor_batch(batch, iteration)
                all_stats.append(stats)

            self.target_accumulated_steps += 1
            if self.target_accumulated_steps % self.cfg.accumulation_steps == 0:
                self.optimizer_target.step()
                self.optimizer_target.zero_grad(set_to_none=True)

        if self.target_accumulated_steps % self.cfg.accumulation_steps != 0:
            self.optimizer_target.step()
            self.optimizer_target.zero_grad(set_to_none=True)

        return self._average_metric_dicts(all_stats, prefix="actor/")

    def _train_actor_batch(self, records: Sequence[TrainingRecord], iteration: int) -> Dict[str, float]:
        input_ids, attention_mask = self._pad_right_with_attention(
            [record.full_input_ids for record in records],
            self.tokenizer.pad_token_id or 0,
        )
        new_token_mask = self._pad_right([record.new_token_mask for record in records], 0, dtype=self.torch.float32)
        response_mask = new_token_mask[:, 1:].to(dtype=self.torch.float32)
        advantages = self._pad_right(
            [[record.advantage * float(mask) for mask in record.new_token_mask[1:]] for record in records],
            0,
            dtype=self.torch.float32,
        )

        outputs = self.model.target_model(input_ids=input_ids, attention_mask=attention_mask)
        log_prob = gather_token_logps(outputs.logits, input_ids)
        needs_reference = self.cfg.objective == "fastgrpo" and self.cfg.beta > 0

        if iteration == 0:
            old_log_prob = log_prob.detach()
            for row, record in enumerate(records):
                seq_len = len(record.full_input_ids) - 1
                record.old_log_prob = old_log_prob[row, :seq_len].detach().cpu()
            ref_log_prob = self._compute_ref_log_prob(input_ids, attention_mask) if needs_reference else None
            if ref_log_prob is not None:
                for row, record in enumerate(records):
                    seq_len = len(record.full_input_ids) - 1
                    record.ref_log_prob = ref_log_prob[row, :seq_len].detach().cpu()
        else:
            old_log_prob = self._pad_logprob_records(records, "old_log_prob")
            ref_log_prob = self._pad_logprob_records(records, "ref_log_prob") if needs_reference else None

        loss_stats = compute_policy_loss(
            objective=self.cfg.objective,
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            ref_log_prob=ref_log_prob,
            advantages=advantages,
            response_mask=response_mask,
            epsilon=self.cfg.epsilon,
            beta=self.cfg.beta,
            loss_agg_mode=self.cfg.loss_agg_mode,
        )
        loss = loss_stats.loss / max(self.cfg.accumulation_steps, 1)
        if self.torch.isnan(loss).any() or self.torch.isinf(loss).any():
            raise FloatingPointError("Actor loss became NaN/Inf")
        loss.backward()
        return {
            "loss": float(loss_stats.loss.detach().cpu()),
            "pg_loss": float(loss_stats.pg_loss.detach().cpu()),
            "kl_loss": float(loss_stats.kl_loss.detach().cpu()),
            "approx_kl": float(loss_stats.approx_kl.detach().cpu()),
            "clip_fraction": float(loss_stats.clip_fraction.detach().cpu()),
            "advantage_mean": float(loss_stats.advantage_mean.detach().cpu()),
            "advantage_abs_mean": float(loss_stats.advantage_abs_mean.detach().cpu()),
        }

    def _compute_ref_log_prob(self, input_ids, attention_mask):
        with self._adapter_disabled(), self.torch.no_grad():
            outputs = self.model.target_model(input_ids=input_ids, attention_mask=attention_mask)
            return gather_token_logps(outputs.logits, input_ids).detach()

    @contextmanager
    def _adapter_disabled(self):
        model = self.model.target_model
        if hasattr(model, "disable_adapter_layers") and hasattr(model, "enable_adapter_layers"):
            model.disable_adapter_layers()
            try:
                yield
            finally:
                model.enable_adapter_layers()
        else:
            yield

    def _pad_logprob_records(self, records: Sequence[TrainingRecord], attr: str):
        rows = []
        max_len = max(len(record.full_input_ids) - 1 for record in records)
        for record in records:
            value = getattr(record, attr)
            if value is None:
                rows.append(self.torch.zeros(max_len, device=self.device))
                continue
            value = value.to(self.device)
            rows.append(self.torch.cat([value, self.torch.zeros(max_len - value.numel(), device=self.device)]))
        return self.torch.stack(rows, dim=0)

    def _iter_packed_records(self, records: Sequence[TrainingRecord]) -> Iterator[List[TrainingRecord]]:
        sorted_records = sorted(records, key=lambda record: len(record.full_input_ids))
        pack: List[TrainingRecord] = []
        cur_max_len = 0
        for record in sorted_records:
            length = len(record.full_input_ids)
            can_add = (
                (max(cur_max_len, length) * (len(pack) + 1) <= self.cfg.max_training_token)
                and ((length - cur_max_len) * len(pack) <= self.cfg.max_training_padding_gap)
            )
            if pack and not can_add:
                yield pack
                pack = []
                cur_max_len = 0
            pack.append(record)
            cur_max_len = max(cur_max_len, length)
        if pack:
            yield pack

    def _train_draft_model(self, outputs: Dict[str, Any], prompts: Sequence[EncodedPrompt]) -> Dict[str, float]:
        self.model.draft_model.train()
        states = outputs.get("all_draft_input_states")
        ids = outputs.get("all_draft_input_ids")
        # Never boolean-coerce tensor-like containers: real or mocked tensors
        # reject ambiguous truth-value checks when they contain multiple items.
        if states is None or ids is None or len(states) == 0 or len(ids) == 0:
            return {"draft/loss1": 0.0, "draft/loss2": 0.0}

        repeat_count = self.cfg.repeated_generate_nums or 1
        prompt_lens = [len(prompts[idx // repeat_count].input_ids) for idx in range(len(states))]
        pairs = sorted(zip(ids, states, prompt_lens), key=lambda x: x[0].shape[-1])

        total_loss1 = 0.0
        total_loss2 = 0.0
        for pack in self._iter_draft_packs(pairs):
            loss1, loss2 = self._train_draft_pack(pack, total_items=len(states))
            total_loss1 += loss1
            total_loss2 += loss2

        self.draft_accumulated_steps += 1
        if self.optimizer_draft is not None and self.draft_accumulated_steps % self.cfg.draft_accumulation_steps == 0:
            self.optimizer_draft.step()
            self.optimizer_draft.zero_grad(set_to_none=True)

        denom = max(len(states), 1)
        return {"draft/loss1": total_loss1 / denom, "draft/loss2": total_loss2 / denom}

    def _iter_draft_packs(self, pairs):
        pack = []
        cur_max_len = 0
        for draft_input_ids, draft_input_states, prompt_len in pairs:
            length = draft_input_ids.shape[-1]
            can_add = (
                (length * (len(pack) + 1) <= self.cfg.max_training_token * 2)
                and ((length - cur_max_len) * len(pack) <= self.cfg.max_training_padding_gap)
            )
            if pack and not can_add:
                yield pack
                pack = []
                cur_max_len = 0
            pack.append((draft_input_ids, draft_input_states, prompt_len))
            cur_max_len = max(cur_max_len, length)
        if pack:
            yield pack

    def _train_draft_pack(self, pack, total_items: int) -> Tuple[float, float]:
        torch = self.torch
        hidden_size = pack[0][1].shape[-1]
        max_len = max(item[0].shape[-1] for item in pack)
        draft_states = []
        draft_ids = []
        loss_masks = []
        attention_masks = []
        for draft_input_ids, draft_input_states, prompt_len in pack:
            length = draft_input_ids.shape[-1]
            pad_len = max_len - length
            if pad_len:
                draft_input_states = torch.cat(
                    [
                        draft_input_states,
                        torch.zeros((pad_len, hidden_size), dtype=draft_input_states.dtype, device=self.device),
                    ],
                    dim=0,
                )
                draft_input_ids = torch.cat(
                    [draft_input_ids, torch.zeros(pad_len, dtype=draft_input_ids.dtype, device=self.device)],
                    dim=0,
                )
            draft_states.append(draft_input_states)
            draft_ids.append(draft_input_ids)
            loss_masks.append([0] * prompt_len + [1] * max(length - prompt_len, 0) + [0] * pad_len)
            attention_masks.append([1] * length + [0] * pad_len)

        draft_states_t = torch.stack(draft_states, dim=0)
        draft_ids_t = torch.stack(draft_ids, dim=0).to(self.device)
        loss_mask_t = torch.tensor(loss_masks, device=self.device, dtype=torch.float32)
        attention_mask_t = torch.tensor(attention_masks, device=self.device, dtype=torch.long)
        l1_loss = torch.nn.SmoothL1Loss(reduction="none")
        autocast_dtype = torch.bfloat16 if self.model.dtype == torch.bfloat16 else torch.float16

        with torch.amp.autocast(self.device.type, dtype=autocast_dtype, enabled=self.device.type == "cuda"):
            draft_outputs = self.model(
                hidden_states=draft_states_t.to(self.model.dtype),
                input_ids=draft_ids_t,
                attention_mask=attention_mask_t,
                use_cache=False,
            )
        next_feature_states = draft_outputs["next_feature_states"]
        draft_hidden_states = draft_outputs["hidden_states"].to(self.model.target_model.dtype)
        draft_logits = self.model.lm_head(draft_hidden_states)

        with torch.no_grad():
            target_logits = self.model.target_model.lm_head(draft_states_t.to(self.model.target_model.dtype))
            target_probs = target_logits[:, 1:, :].float().softmax(dim=-1).detach()

        token_mask = loss_mask_t[:, :-1]
        denom = token_mask.sum(dim=-1).clamp_min(1.0)
        loss1 = l1_loss(next_feature_states[:, :-1, :].float(), draft_states_t[:, 1:, :].float())
        loss1 = (loss1.mean(dim=-1) * token_mask).sum(dim=-1) / denom
        loss1 = loss1.sum() * 2.0

        draft_probs = draft_logits[:, :-1, :].float().softmax(dim=-1).clamp_min(1e-12)
        loss2 = -(target_probs * draft_probs.log()).sum(dim=-1)
        loss2 = (loss2 * token_mask).sum(dim=-1) / denom
        loss2 = loss2.sum() * 0.1

        loss = (loss1 + loss2) / max(total_items, 1) / max(self.cfg.draft_accumulation_steps, 1)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Draft loss became NaN/Inf")
        loss.backward()
        return float(loss1.detach().cpu()), float(loss2.detach().cpu())

    def _average_metric_dicts(self, metrics: Sequence[Dict[str, float]], prefix: str = "") -> Dict[str, float]:
        if not metrics:
            return {}
        keys = sorted({key for metric in metrics for key in metric})
        return {prefix + key: float(np.mean([metric[key] for metric in metrics if key in metric])) for key in keys}

    def _save_checkpoint(self, step: int) -> None:
        target_dir = Path(self.cfg.saved_model_dir) / f"step{step}"
        draft_path = Path(self.cfg.saved_draft_model_dir) / f"step{step}.pth"
        target_dir.mkdir(parents=True, exist_ok=True)
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.target_model.save_pretrained(target_dir)
        self.tokenizer.save_pretrained(target_dir)
        if self.cfg.generation_backend == "speculative" and hasattr(self.model, "save_model"):
            self.model.save_model(str(draft_path))
