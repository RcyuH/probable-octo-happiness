"""FastGRPO-based PROS trainer."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .pros_config import ProsConfig
from .pros_loss import compute_gpg_advantages, compute_grpo_advantages, compute_policy_loss, gather_token_logps
from .pros_tree import ProsRolloutRecord, ProsTreeConfig, ProsTreeEngine


@dataclass
class EncodedPrompt:
    item: int
    ancestor_item: int
    question: str
    answer: str
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
    advantage: float = 0.0
    old_log_prob: Any = None
    ref_log_prob: Any = None
    entropies: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    @property
    def response_mask_for_tree(self) -> List[int]:
        return [1] * len(self.full_response_ids)


class ProsTrainer:
    """Orchestrates PROS training while reusing FastGRPO helpers."""

    def __init__(self, config: ProsConfig):
        self.cfg = config
        self.repo_root = Path(__file__).resolve().parents[2]
        self.fastgrpo_root = self.repo_root / "FastGRPO"
        self._ensure_fastgrpo_importable()

        self.torch = self._import_torch()
        self._seed_everything(config.seed)
        self.device = self._resolve_device()

        self._load_fastgrpo_helpers()
        self._load_models()
        self.qas = self.get_train_QAs(config.train_option)
        self.eval_qas = self.get_test_QAs(config.eval_option) if config.eval_option else []
        self.tree = ProsTreeEngine(len(self.qas), self._tree_config())
        self.next_items, self.last_sampler_metrics = self.tree.select_batch(config.batch_size, step_num=0)

        self.optimizer_target = self.torch.optim.AdamW(
            [p for p in self.model.target_model.parameters() if p.requires_grad],
            lr=config.target_lr,
        )
        self.optimizer_draft = (
            self.torch.optim.AdamW(self.model.draft_model.parameters(), lr=config.draft_lr) if config.train_draft else None
        )
        self.target_accumulated_steps = 0
        self.draft_accumulated_steps = 0
        self.global_step = 0
        self.used_groups = 0
        self.start_time = time.time()

        self._prepare_output_dirs()
        self.cfg.save()

    def train(self) -> None:
        if self.cfg.dry_run:
            print(json.dumps({"dry_run": True, "config": self.cfg.to_dict()}, indent=2, sort_keys=True))
            return

        for epoch in range(self.cfg.num_epochs):
            if self.cfg.max_train_steps and self.global_step >= self.cfg.max_train_steps:
                break
            self._train_epoch(epoch)

        self._save_checkpoint(self.global_step)

    def _train_epoch(self, epoch: int) -> None:
        while True:
            if self.cfg.max_train_steps and self.global_step >= self.cfg.max_train_steps:
                return
            prompts = self._encode_selected_items(self.next_items)
            if not prompts:
                self.next_items, self.last_sampler_metrics = self.tree.select_batch(self.cfg.batch_size, self.global_step)
                continue

            generation_start = time.time()
            outputs = self._generate(prompts)
            records = self._build_training_records(prompts, outputs)
            if self.cfg.train_draft and self.optimizer_draft is not None:
                draft_metrics = self._train_draft_model(outputs, prompts)
            else:
                draft_metrics = {}

            if not records:
                self.next_items, self.last_sampler_metrics = self.tree.select_batch(self.cfg.batch_size, self.global_step)
                continue

            self._attach_response_statistics(records)
            self._attach_advantages(records)
            train_metrics = self._train_actor(records)

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
                for record in records
            ]
            self.next_items, tree_metrics = self.tree.update_and_select(
                tree_records,
                step_num=self.global_step + 1,
                batch_size=self.cfg.batch_size,
            )

            token_lengths = [len(record.generated_ids) for record in records]
            reward_values = [record.reward for record in records]
            metrics = {
                "epoch": epoch + 1,
                "step": self.global_step,
                "used_groups": self.used_groups,
                "records": len(records),
                "elapsed_min": round((time.time() - self.start_time) / 60.0, 4),
                "generate_time_sec": round(time.time() - generation_start, 4),
                "reward_mean": float(np.mean(reward_values)) if reward_values else 0.0,
                "reward_std": float(np.std(reward_values)) if reward_values else 0.0,
                "generated_len_mean": float(np.mean(token_lengths)) if token_lengths else 0.0,
                "generated_len_std": float(np.std(token_lengths)) if len(token_lengths) > 1 else 0.0,
                **self.last_sampler_metrics,
                **tree_metrics,
                **train_metrics,
                **draft_metrics,
            }
            if self.cfg.eval_freq > 0 and self.eval_qas and (self.global_step + 1) % self.cfg.eval_freq == 0:
                metrics.update(self._evaluate())
            self._log(metrics)

            self.global_step += 1
            if self.cfg.save_freq > 0 and self.global_step % self.cfg.save_freq == 0:
                self._save_checkpoint(self.global_step)

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
        from helper.rewards import accuracy_reward_func, format_reward_func
        from helper.specualtive_generate import speculative_generate

        self.Model = Model
        self.get_train_QAs = get_train_QAs
        self.get_test_QAs = get_test_QAs
        self.accuracy_reward_func = accuracy_reward_func
        self.format_reward_func = format_reward_func
        self.speculative_generate = speculative_generate

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
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
            prompt_ids = self._encode_base_prompt(qa["question"])
            partial = node.partial_rollout or []
            input_ids = prompt_ids + partial
            if len(input_ids) >= self.cfg.max_length:
                continue
            prompts.append(
                EncodedPrompt(
                    item=int(item),
                    ancestor_item=ancestor,
                    question=qa["question"],
                    answer=qa["answer"],
                    prompt_ids=prompt_ids,
                    partial_rollout=partial,
                    input_ids=input_ids,
                )
            )
        return prompts

    def _encode_base_prompt(self, question: str) -> List[int]:
        system_prompt = "You are a math problem assistant."
        user_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "Write a response that appropriately completes the request.\n"
            "Your response should include your thought process enclosed within <think></think> tags\n"
            "and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n\n"
            f"### Instruction:\n{question}\n"
            "Please reason step by step, and put your final answer within \\boxed{}"
        )
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
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
        input_ids, attention_mask = self._pad_left([prompt.input_ids for prompt in prompts])
        statistical_time = bool(self.torch.cuda.is_available())
        top_k = self.cfg.top_k if self.cfg.top_k > 0 else None
        with self.torch.inference_mode():
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
                return_all_draft_input=self.cfg.train_draft,
                statistical_time=statistical_time,
            )

    def _evaluate(self) -> Dict[str, float]:
        eval_examples = self.eval_qas[: self.cfg.eval_samples]
        if not eval_examples:
            return {}

        self.model.target_model.eval()
        prompts = [
            EncodedPrompt(
                item=i,
                ancestor_item=i,
                question=example["question"],
                answer=example["answer"],
                prompt_ids=self._encode_base_prompt(example["question"]),
                partial_rollout=[],
                input_ids=self._encode_base_prompt(example["question"]),
            )
            for i, example in enumerate(eval_examples)
        ]
        rewards: List[float] = []
        accuracies: List[float] = []
        for start in range(0, len(prompts), self.cfg.batch_size):
            batch_prompts = prompts[start : start + self.cfg.batch_size]
            input_ids, attention_mask = self._pad_left([prompt.input_ids for prompt in batch_prompts])
            with self.torch.inference_mode():
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
                    top_k=self.cfg.top_k if self.cfg.top_k > 0 else None,
                    return_all_draft_input=False,
                    statistical_time=bool(self.torch.cuda.is_available()),
                )
            for prompt, generated_ids in zip(batch_prompts, outputs["generated_token_ids"]):
                completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                format_reward = self.format_reward_func([completion])[0]
                answer_reward = self.accuracy_reward_func([completion], [prompt.answer])[0]
                rewards.append(float(0.2 * format_reward + answer_reward))
                accuracies.append(float(answer_reward))
        return {
            "eval/reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "eval/accuracy_mean": float(np.mean(accuracies)) if accuracies else 0.0,
            "eval/samples": float(len(rewards)),
        }

    def _build_training_records(self, prompts: Sequence[EncodedPrompt], outputs: Dict[str, Any]) -> List[TrainingRecord]:
        generated = outputs["generated_token_ids"]
        decoded_by_sample = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
        prompt_records: Dict[int, List[TrainingRecord]] = {}

        for idx, generated_ids in enumerate(generated):
            prompt_idx = idx // self.cfg.repeated_generate_nums
            prompt = prompts[prompt_idx]
            generated_ids = list(generated_ids)
            if not generated_ids:
                continue
            full_response_ids = prompt.partial_rollout + generated_ids
            decoded_completion = self.tokenizer.decode(full_response_ids, skip_special_tokens=True)
            format_reward = self.format_reward_func([decoded_completion])[0]
            answer_reward = self.accuracy_reward_func([decoded_completion], [prompt.answer])[0]
            reward = float(0.2 * format_reward + answer_reward)
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
                )
            )

        records: List[TrainingRecord] = []
        for group in prompt_records.values():
            rewards = [record.reward for record in group]
            if self.cfg.drop_zero_std_groups and len(group) > 1 and float(np.std(rewards)) == 0.0:
                if rewards and rewards[0] >= self.cfg.tree_score_threshold:
                    pass
                continue
            records.extend(group)
            self.used_groups += 1
        return records

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

        if iteration == 0:
            old_log_prob = log_prob.detach()
            for row, record in enumerate(records):
                seq_len = len(record.full_input_ids) - 1
                record.old_log_prob = old_log_prob[row, :seq_len].detach().cpu()
            ref_log_prob = self._compute_ref_log_prob(input_ids, attention_mask) if self.cfg.beta > 0 else None
            if ref_log_prob is not None:
                for row, record in enumerate(records):
                    seq_len = len(record.full_input_ids) - 1
                    record.ref_log_prob = ref_log_prob[row, :seq_len].detach().cpu()
        else:
            old_log_prob = self._pad_logprob_records(records, "old_log_prob")
            ref_log_prob = self._pad_logprob_records(records, "ref_log_prob") if self.cfg.beta > 0 else None

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
        states = outputs.get("all_draft_input_states") or []
        ids = outputs.get("all_draft_input_ids") or []
        if not states or not ids:
            return {"draft/loss1": 0.0, "draft/loss2": 0.0}

        prompt_lens = [len(prompts[idx // self.cfg.repeated_generate_nums].input_ids) for idx in range(len(states))]
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

    def _log(self, metrics: Dict[str, Any]) -> None:
        with open(self.cfg.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        if self.cfg.log_freq > 0 and self.global_step % self.cfg.log_freq == 0:
            printable = {k: v for k, v in metrics.items() if not isinstance(v, list)}
            print(json.dumps(printable, sort_keys=True))

    def _save_checkpoint(self, step: int) -> None:
        target_dir = Path(self.cfg.saved_model_dir) / f"step{step}"
        draft_path = Path(self.cfg.saved_draft_model_dir) / f"step{step}.pth"
        target_dir.mkdir(parents=True, exist_ok=True)
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.target_model.save_pretrained(target_dir)
        self.tokenizer.save_pretrained(target_dir)
        if hasattr(self.model, "save_model"):
            self.model.save_model(str(draft_path))
