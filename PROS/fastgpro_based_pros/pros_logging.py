"""Dependency-light metrics and event logging for the PROS trainer.

This module intentionally does not import NumPy, Torch, or TensorBoard at
module import time.  Training code can therefore reuse and test the metric
formulas without loading a model runtime.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import warnings
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import stdev
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence


JsonDict = Dict[str, Any]


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, or a finite default."""

    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        converted = float(default)
    if not math.isfinite(converted):
        converted = float(default)
    return converted if math.isfinite(converted) else 0.0


def safe_div(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """Divide two numeric values without ever returning NaN or infinity."""

    fallback = _finite_float(default)
    numerator_value = _finite_float(numerator, fallback)
    denominator_value = _finite_float(denominator, 0.0)
    if denominator_value == 0.0:
        return fallback
    result = numerator_value / denominator_value
    return result if math.isfinite(result) else fallback


safe_divide = safe_div


def to_json_safe(value: Any) -> Any:
    """Recursively convert common training values into strict JSON values.

    NumPy and Torch values are supported through their public ``item`` and
    ``tolist`` protocols, without importing either dependency.  Non-finite
    floating-point values are converted to ``0.0`` so callers can serialize
    with ``allow_nan=False``.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [to_json_safe(item) for item in sorted(value, key=repr)]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item = item_method()
        except (TypeError, ValueError, RuntimeError):
            item = value
        if item is not value:
            return to_json_safe(item)

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            listed = tolist_method()
        except (TypeError, ValueError, RuntimeError):
            listed = value
        if listed is not value:
            return to_json_safe(listed)

    return str(value)


def _numeric_stats(values: Iterable[Any], prefix: str) -> JsonDict:
    finite_values: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            finite_values.append(number)

    count = len(finite_values)
    if not finite_values:
        mean_value = min_value = max_value = stdev_value = 0.0
    else:
        mean_value = sum(finite_values) / count
        min_value = min(finite_values)
        max_value = max(finite_values)
        stdev_value = stdev(finite_values) if count > 1 else 0.0
    range_value = max_value - min_value
    return {
        f"{prefix}_count": count,
        f"{prefix}_mean": _finite_float(mean_value),
        f"{prefix}_min": _finite_float(min_value),
        f"{prefix}_max": _finite_float(max_value),
        f"{prefix}_stdev": _finite_float(stdev_value),
        f"{prefix}_range": _finite_float(range_value),
        f"{prefix}_cv": safe_div(stdev_value, mean_value),
    }


def compute_length_metrics(
    suffix_lengths: Iterable[Any],
    partial_lengths: Iterable[Any] = (),
    full_lengths: Iterable[Any] = (),
) -> JsonDict:
    """Summarize generated suffix, inherited partial, and full response lengths.

    The standard deviation is the sample standard deviation (``n - 1``), as
    requested by the training log contract.  Empty and singleton inputs
    produce finite zero-valued statistics.
    """

    metrics: JsonDict = {}
    metrics.update(_numeric_stats(suffix_lengths, "suffix_token_length"))
    metrics.update(_numeric_stats(partial_lengths, "inherited_partial_rollout_length"))
    metrics.update(_numeric_stats(full_lengths, "full_response_length"))
    return metrics


length_metrics = compute_length_metrics


def _nonnegative_int(value: Any) -> int:
    return max(int(_finite_float(value)), 0)


def _nonnegative_float(value: Any) -> float:
    return max(_finite_float(value), 0.0)


def compute_generation_perf(
    outputs: Mapping[str, Any],
    suffix_lengths: Optional[Iterable[Any]] = None,
    *,
    generation_time_sec: Optional[float] = None,
    generation_backend: str = "speculative",
) -> JsonDict:
    """Compute finite generation and speculative-decoding performance metrics.

    ``generation_time_sec`` should be the wall-clock duration measured only
    around generation.  If omitted, the generator's ``total_time_cost`` is
    used.  Target-only generation intentionally emits the same schema while
    forcing every speculative counter and derived rate to zero.
    """

    backend = str(generation_backend).strip().lower()
    if backend not in {"speculative", "target"}:
        raise ValueError(f"Unknown generation backend: {generation_backend!r}")

    if suffix_lengths is None:
        generated_rows = outputs.get("generated_token_ids", ())
        suffix_values = [len(row) for row in generated_rows]
    else:
        suffix_values = list(suffix_lengths)
    generated_tokens = sum(_nonnegative_int(value) for value in suffix_values)

    reported_time = _nonnegative_float(outputs.get("total_time_cost", 0.0))
    wall_time = reported_time if generation_time_sec is None else _nonnegative_float(generation_time_sec)
    target_time = _nonnegative_float(outputs.get("target_time_cost", 0.0))
    draft_time = _nonnegative_float(outputs.get("draft_time_cost", 0.0))
    check_time = _nonnegative_float(outputs.get("check_time_cost", 0.0))
    prefill_time = _nonnegative_float(outputs.get("prefill_time_cost", 0.0))

    if backend == "target":
        emitted_tokens = accepted_tokens = verified_tokens = path_budget_tokens = rounds = 0
    else:
        emitted_tokens = _nonnegative_int(
            outputs.get("speculative_emitted_tokens", outputs.get("total_acc_length", 0))
        )
        accepted_tokens = _nonnegative_int(outputs.get("speculative_accepted_draft_tokens", 0))
        verified_tokens = _nonnegative_int(outputs.get("speculative_verified_draft_tokens", 0))
        path_budget_tokens = _nonnegative_int(outputs.get("speculative_path_budget_tokens", 0))
        rounds = _nonnegative_int(
            outputs.get("speculative_verification_rounds", outputs.get("total_decoded_token_num", 0))
        )

    return {
        "generation_backend": backend,
        "generated_completion_tokens": generated_tokens,
        "generated_new_tokens": generated_tokens,
        "generation_time_sec": wall_time,
        "reported_generation_time_sec": reported_time,
        "generated_tokens_per_second": safe_div(generated_tokens, wall_time),
        "target_time_sec": target_time,
        "draft_time_sec": draft_time,
        "check_time_sec": check_time,
        "prefill_time_sec": prefill_time,
        "target_time_ratio": safe_div(target_time, wall_time),
        "draft_time_ratio": safe_div(draft_time, wall_time),
        "check_time_ratio": safe_div(check_time, wall_time),
        "prefill_time_ratio": safe_div(prefill_time, wall_time),
        "speculative_verification_rounds": rounds,
        "speculative_emitted_tokens": emitted_tokens,
        "speculative_accepted_draft_tokens": accepted_tokens,
        "speculative_verified_draft_tokens": verified_tokens,
        "speculative_path_budget_tokens": path_budget_tokens,
        "speculative_avg_emitted_tokens_per_round": safe_div(emitted_tokens, rounds),
        "speculative_avg_accepted_draft_tokens_per_round": safe_div(accepted_tokens, rounds),
        "speculative_path_acceptance_rate": safe_div(accepted_tokens, path_budget_tokens),
        "speculative_tree_acceptance_rate": safe_div(accepted_tokens, verified_tokens),
        "speculative_verified_draft_tokens_per_round": safe_div(verified_tokens, rounds),
    }


generation_perf_metrics = compute_generation_perf


def new_reward_debug_stats() -> JsonDict:
    """Create mutable accumulator state for reward diagnostics."""

    return {
        "completion_count": 0,
        "reward_sum": 0.0,
        "reward_sq_sum": 0.0,
        "pass_count": 0,
        "fail_count": 0,
        "timeout_count": 0,
        "missing_tests_count": 0,
        "missing_entry_point_count": 0,
        "answer_parse_failed_count": 0,
        "gold_parse_failed_count": 0,
        "answer_parsed_count": 0,
        "gold_parsed_count": 0,
        "answer_tag_fallback_count": 0,
        "verify_error_count": 0,
        "format_reward_sum": 0.0,
        "format_reward_count": 0,
        "completion_chars_sum": 0.0,
        "extracted_code_chars_sum": 0.0,
        "stdout_chars_sum": 0.0,
        "stderr_chars_sum": 0.0,
        "used_group_count": 0,
        "skip_group_count": 0,
        "skip_due_correct_group_count": 0,
        "skip_due_incorrect_group_count": 0,
        "reward_type_counts": {},
        "error_type_counts": {},
        "test_type_counts": {},
        "ignored_correct_error_type_counts": {},
        "ignored_incorrect_error_type_counts": {},
    }


def _counter_key(value: Any, default: str = "none") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _increment_counter(counter: MutableMapping[str, int], key: Any, amount: int = 1) -> None:
    counter_key = _counter_key(key)
    counter[counter_key] = int(counter.get(counter_key, 0)) + int(amount)


def _detail_passed(detail: Mapping[str, Any], reward: Optional[float] = None) -> bool:
    if "passed" in detail and detail.get("passed") is not None:
        return detail.get("passed") is True
    reward_value = _finite_float(detail.get("reward", 0.0) if reward is None else reward)
    return reward_value >= 1.0


def record_reward_detail(stats: MutableMapping[str, Any], detail: Mapping[str, Any]) -> None:
    """Record one already-evaluated completion in ``stats``.

    This helper only consumes reward details.  It never calls a verifier, so
    using it cannot accidentally execute a custom/code reward twice.
    """

    reward = _finite_float(detail.get("reward", 0.0))
    stats["completion_count"] += 1
    stats["reward_sum"] += reward
    stats["reward_sq_sum"] += reward * reward
    if _detail_passed(detail, reward):
        stats["pass_count"] += 1
    else:
        stats["fail_count"] += 1

    error_type = _counter_key(detail.get("error_type"))
    if detail.get("timed_out") is True or error_type.lower() == "timeout":
        stats["timeout_count"] += 1
    if detail.get("has_tests") is False:
        stats["missing_tests_count"] += 1

    test_type = str(detail.get("test_type") or "").lower()
    stdio_types = {"stdin_stdout", "stdio", "io", "input_output"}
    if (
        detail.get("has_entry_point") is False
        and detail.get("reward_type") == "code_unit_test"
        and test_type not in stdio_types
    ):
        stats["missing_entry_point_count"] += 1

    if detail.get("answer_parse_failed") is True:
        stats["answer_parse_failed_count"] += 1
    if detail.get("gold_parse_failed") is True:
        stats["gold_parse_failed_count"] += 1

    # The shared math verifier returns integer parse counts plus bounded text
    # representations.  Older/custom verifiers may instead return booleans,
    # so retain that form as a compatibility fallback.
    if detail.get("answer_parsed_count") is not None:
        stats["answer_parsed_count"] += _nonnegative_int(detail.get("answer_parsed_count"))
    elif detail.get("answer_parsed") is True:
        stats["answer_parsed_count"] += 1
    if detail.get("gold_parsed_count") is not None:
        stats["gold_parsed_count"] += _nonnegative_int(detail.get("gold_parsed_count"))
    elif detail.get("gold_parsed") is True:
        stats["gold_parsed_count"] += 1
    if detail.get("answer_tag_fallback_used") is True:
        stats["answer_tag_fallback_count"] += 1
    if detail.get("verify_error") not in (None, "", False):
        stats["verify_error_count"] += 1
    if detail.get("format_reward") is not None:
        stats["format_reward_sum"] += _finite_float(detail.get("format_reward"))
        stats["format_reward_count"] += 1

    for field_name in ("completion_chars", "extracted_code_chars", "stdout_chars", "stderr_chars"):
        stats[f"{field_name}_sum"] += _nonnegative_float(detail.get(field_name, 0.0))

    _increment_counter(stats["reward_type_counts"], detail.get("reward_type"), 1)
    _increment_counter(stats["error_type_counts"], detail.get("error_type"), 1)
    if detail.get("test_type") not in (None, ""):
        _increment_counter(stats["test_type_counts"], detail.get("test_type"), 1)


_USED_DECISIONS = {"used", "use", "actor_eligible"}
_CORRECT_SKIP_DECISIONS = {
    "ignore_due_correct",
    "skip_due_correct",
    "skipped_correct",
    "ignored_correct",
}
_INCORRECT_SKIP_DECISIONS = {
    "ignore_due_incorrect",
    "skip_due_incorrect",
    "skipped_incorrect",
    "ignored_incorrect",
}


def record_group_decision(
    stats: MutableMapping[str, Any],
    decision: str,
    reward_details: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Record whether one rollout group was used or skipped."""

    normalized = str(decision).strip().lower()
    if normalized in _USED_DECISIONS:
        stats["used_group_count"] += 1
        return
    if normalized in _CORRECT_SKIP_DECISIONS:
        stats["skip_group_count"] += 1
        stats["skip_due_correct_group_count"] += 1
        error_counter = stats["ignored_correct_error_type_counts"]
    elif normalized in _INCORRECT_SKIP_DECISIONS:
        stats["skip_group_count"] += 1
        stats["skip_due_incorrect_group_count"] += 1
        error_counter = stats["ignored_incorrect_error_type_counts"]
    else:
        raise ValueError(f"Unknown rollout group decision: {decision!r}")

    for detail in reward_details:
        _increment_counter(error_counter, detail.get("error_type"), 1)


def _sorted_counter(counter: Mapping[str, Any]) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def summarize_reward_debug(stats: Mapping[str, Any]) -> JsonDict:
    """Return strict-JSON-safe reward diagnostics from accumulator state."""

    count = int(stats.get("completion_count", 0))
    reward_mean = safe_div(stats.get("reward_sum", 0.0), count)
    reward_variance = safe_div(stats.get("reward_sq_sum", 0.0), count) - reward_mean * reward_mean
    reward_variance = max(_finite_float(reward_variance), 0.0)
    format_reward_count = int(stats.get("format_reward_count", 0))
    return {
        "completion_count": count,
        "mean_reward_all_completions": reward_mean,
        "reward_std_all_completions": math.sqrt(reward_variance),
        "pass_count": int(stats.get("pass_count", 0)),
        "fail_count": int(stats.get("fail_count", 0)),
        "pass_rate": safe_div(stats.get("pass_count", 0), count),
        "fail_rate": safe_div(stats.get("fail_count", 0), count),
        "timeout_count": int(stats.get("timeout_count", 0)),
        "missing_tests_count": int(stats.get("missing_tests_count", 0)),
        "missing_entry_point_count": int(stats.get("missing_entry_point_count", 0)),
        "answer_parse_failed_count": int(stats.get("answer_parse_failed_count", 0)),
        "gold_parse_failed_count": int(stats.get("gold_parse_failed_count", 0)),
        "answer_parsed_count": int(stats.get("answer_parsed_count", 0)),
        "gold_parsed_count": int(stats.get("gold_parsed_count", 0)),
        "answer_tag_fallback_count": int(stats.get("answer_tag_fallback_count", 0)),
        "verify_error_count": int(stats.get("verify_error_count", 0)),
        "mean_format_reward": safe_div(stats.get("format_reward_sum", 0.0), format_reward_count),
        "mean_completion_chars": safe_div(stats.get("completion_chars_sum", 0.0), count),
        "mean_extracted_code_chars": safe_div(stats.get("extracted_code_chars_sum", 0.0), count),
        "mean_stdout_chars": safe_div(stats.get("stdout_chars_sum", 0.0), count),
        "mean_stderr_chars": safe_div(stats.get("stderr_chars_sum", 0.0), count),
        "used_group_count": int(stats.get("used_group_count", 0)),
        "skip_group_count": int(stats.get("skip_group_count", 0)),
        "skip_due_correct_group_count": int(stats.get("skip_due_correct_group_count", 0)),
        "skip_due_incorrect_group_count": int(stats.get("skip_due_incorrect_group_count", 0)),
        "reward_type_counts": _sorted_counter(stats.get("reward_type_counts", {})),
        "error_type_counts": _sorted_counter(stats.get("error_type_counts", {})),
        "test_type_counts": _sorted_counter(stats.get("test_type_counts", {})),
        "ignored_correct_error_type_counts": _sorted_counter(
            stats.get("ignored_correct_error_type_counts", {})
        ),
        "ignored_incorrect_error_type_counts": _sorted_counter(
            stats.get("ignored_incorrect_error_type_counts", {})
        ),
    }


def clip_log_text(value: Any, limit: int) -> str:
    """Stringify and truncate a value to at most ``limit`` characters."""

    text = str(value or "")
    limit = max(int(limit), 0)
    if len(text) <= limit:
        return text
    marker = "...<truncated>"
    if limit <= len(marker):
        return marker[:limit]
    return text[: limit - len(marker)] + marker


def _compact_prompt(prompt: Any, limit: int) -> str:
    if isinstance(prompt, Mapping) and "messages" in prompt:
        prompt = prompt.get("messages")
    if isinstance(prompt, (list, tuple)):
        pieces = []
        for message in prompt:
            if isinstance(message, Mapping):
                pieces.append(f"{message.get('role', 'unknown')}: {message.get('content', '')}")
            else:
                pieces.append(str(message))
        return clip_log_text("\n".join(pieces), limit)
    if isinstance(prompt, Mapping):
        return clip_log_text(json.dumps(to_json_safe(prompt), sort_keys=True), limit)
    return clip_log_text(prompt, limit)


_FAILED_SAMPLE_DETAIL_KEYS = (
    "answer_parse_failed",
    "gold_parse_failed",
    "answer_tag_fallback_used",
    "answer_parsed_count",
    "gold_parsed_count",
    "answer_parsed",
    "gold_parsed",
    "format_reward",
    "verify_error",
    "test_type",
    "has_tests",
    "has_entry_point",
    "timed_out",
    "stderr_excerpt",
    "stdout_excerpt",
    "stderr",
    "stdout",
)


def make_failed_reward_sample(
    detail: Mapping[str, Any],
    *,
    task_id: str,
    prompt: Any,
    completion: Any,
    char_limit: int,
    repeat_index: Optional[int] = None,
) -> Optional[JsonDict]:
    """Build a bounded debug sample, or ``None`` for a passing completion."""

    reward = _finite_float(detail.get("reward", 0.0))
    error_type = detail.get("error_type")
    neutral_error = str(error_type or "").strip().lower() in {"", "none", "ok", "success"}
    if _detail_passed(detail, reward) and neutral_error:
        return None

    sample: JsonDict = {
        "task_id": str(task_id),
        "reward": reward,
        "reward_type": detail.get("reward_type"),
        "error_type": error_type,
        "prompt": _compact_prompt(prompt, char_limit),
        "completion": clip_log_text(completion, char_limit),
    }
    if repeat_index is not None:
        sample["repeat_index"] = int(repeat_index)
    for key in _FAILED_SAMPLE_DETAIL_KEYS:
        if key not in detail:
            continue
        item = detail.get(key)
        sample[key] = clip_log_text(item, char_limit) if isinstance(item, str) else to_json_safe(item)
    return sample


class RewardDebugAggregator:
    """Accumulate batch, cumulative, and per-task reward diagnostics."""

    def __init__(self, sample_limit: int = 5, char_limit: int = 800):
        self.sample_limit = max(int(sample_limit), 0)
        self.char_limit = max(int(char_limit), 0)
        self.cumulative_stats = new_reward_debug_stats()
        self.cumulative_per_task: Dict[str, JsonDict] = {}
        self.cumulative_failed_samples: list[JsonDict] = []
        self.reset_batch()

    def reset_batch(self) -> None:
        self.batch_stats = new_reward_debug_stats()
        self.batch_per_task: Dict[str, JsonDict] = {}
        self.batch_failed_samples: list[JsonDict] = []

    @staticmethod
    def _task_stats(store: MutableMapping[str, JsonDict], task_id: str) -> JsonDict:
        key = str(task_id or "default")
        if key not in store:
            store[key] = new_reward_debug_stats()
        return store[key]

    def record_completion(
        self,
        detail: Mapping[str, Any],
        *,
        task_id: str = "default",
        prompt: Any = "",
        completion: Any = "",
        repeat_index: Optional[int] = None,
    ) -> None:
        if not isinstance(detail, Mapping):
            raise TypeError("reward detail must be a mapping")
        normalized_detail = dict(detail)
        normalized_detail["reward"] = _finite_float(normalized_detail.get("reward", 0.0))
        normalized_detail.setdefault("completion_chars", len(str(completion or "")))
        if "stdout_chars" not in normalized_detail and normalized_detail.get("stdout_excerpt") is not None:
            normalized_detail["stdout_chars"] = len(str(normalized_detail.get("stdout_excerpt") or ""))
        if "stderr_chars" not in normalized_detail and normalized_detail.get("stderr_excerpt") is not None:
            normalized_detail["stderr_chars"] = len(str(normalized_detail.get("stderr_excerpt") or ""))

        task_key = str(task_id or "default")
        for stats in (
            self.batch_stats,
            self.cumulative_stats,
            self._task_stats(self.batch_per_task, task_key),
            self._task_stats(self.cumulative_per_task, task_key),
        ):
            record_reward_detail(stats, normalized_detail)

        sample = make_failed_reward_sample(
            normalized_detail,
            task_id=task_key,
            prompt=prompt,
            completion=completion,
            char_limit=self.char_limit,
            repeat_index=repeat_index,
        )
        if sample is not None:
            if len(self.batch_failed_samples) < self.sample_limit:
                self.batch_failed_samples.append(sample)
            if len(self.cumulative_failed_samples) < self.sample_limit:
                self.cumulative_failed_samples.append(sample)

    def record_group_decision(
        self,
        decision: str,
        reward_details: Sequence[Mapping[str, Any]] = (),
        *,
        task_id: str = "default",
    ) -> None:
        task_key = str(task_id or "default")
        for stats in (
            self.batch_stats,
            self.cumulative_stats,
            self._task_stats(self.batch_per_task, task_key),
            self._task_stats(self.cumulative_per_task, task_key),
        ):
            record_group_decision(stats, decision, reward_details)

    def snapshot(self) -> JsonDict:
        return {
            "batch": summarize_reward_debug(self.batch_stats),
            "cumulative": summarize_reward_debug(self.cumulative_stats),
            "per_task_batch": {
                task_id: summarize_reward_debug(stats)
                for task_id, stats in sorted(self.batch_per_task.items())
            },
            "per_task": {
                task_id: summarize_reward_debug(stats)
                for task_id, stats in sorted(self.cumulative_per_task.items())
            },
            "failed_samples": list(self.batch_failed_samples),
            "failed_samples_cumulative": list(self.cumulative_failed_samples),
        }

    def as_payload(self) -> JsonDict:
        snapshot = self.snapshot()
        return {
            "reward_debug_batch": snapshot["batch"],
            "reward_debug": snapshot["cumulative"],
            "reward_debug_per_task_batch": snapshot["per_task_batch"],
            "reward_debug_per_task": snapshot["per_task"],
            "reward_debug_samples": snapshot["failed_samples"],
        }


def sanitize_tb_tag(value: Any) -> str:
    """Return a TensorBoard-safe tag while preserving hierarchy slashes."""

    tag = re.sub(r"[^A-Za-z0-9_./-]", "_", str(value))
    tag = re.sub(r"/{2,}", "/", tag).strip("/")
    return tag or "unknown"


sanitize_tag = sanitize_tb_tag


def _iter_finite_scalars(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 8,
    limit: int = 512,
) -> Iterable[tuple[str, float]]:
    emitted = 0
    stack: list[tuple[str, Any, int]] = [
        (str(key), value, depth) for key, value in reversed(list(values.items()))
    ]
    while stack and emitted < limit:
        path, value, current_depth = stack.pop()
        full_path = f"{prefix}/{path}" if prefix else path
        if isinstance(value, Mapping) and current_depth < max_depth:
            children = list(value.items())
            for child_key, child_value in reversed(children):
                stack.append((f"{path}/{child_key}", child_value, current_depth + 1))
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        emitted += 1
        yield sanitize_tb_tag(full_path), numeric


def write_tensorboard_scalars(
    writer: Any,
    payload: Mapping[str, Any],
    step: int,
    *,
    prefix: str = "",
    max_scalars: int = 512,
) -> None:
    """Write finite scalar leaves only; lists and text are deliberately ignored."""

    if writer is None:
        return
    for tag, value in _iter_finite_scalars(payload, prefix=prefix, limit=max_scalars):
        writer.add_scalar(tag, value, int(step))


def _load_summary_writer_class():
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter


def build_summary_writer(
    *,
    use_tensorboard: bool,
    tensorboard_log_dir: str | os.PathLike[str],
    summary_writer_cls: Optional[Callable[..., Any]] = None,
    warning_fn: Optional[Callable[[str], None]] = None,
) -> Any:
    """Construct an optional SummaryWriter, falling back without failing training."""

    if not use_tensorboard:
        return None
    warn = warning_fn or (lambda message: warnings.warn(message, RuntimeWarning, stacklevel=2))
    try:
        writer_cls = summary_writer_cls or _load_summary_writer_class()
    except (ImportError, ModuleNotFoundError) as exc:
        warn(f"TensorBoard is unavailable ({exc}); continuing with JSONL logging only.")
        return None

    log_dir = Path(tensorboard_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        return writer_cls(log_dir=str(log_dir))
    except Exception as exc:  # TensorBoard is optional; JSONL must remain usable.
        warn(f"TensorBoard writer initialization failed ({exc}); continuing with JSONL logging only.")
        return None


class ProsEventLogger:
    """Append strict JSONL events and mirror finite scalars to TensorBoard.

    The logger owns the monotonically increasing ``tb_step`` across generation,
    train, and evaluation events.  Callers provide event-specific context such
    as ``epoch``, ``global_step``, and ``generation_attempt``.
    """

    def __init__(
        self,
        log_file: str | os.PathLike[str],
        *,
        use_tensorboard: bool = True,
        tensorboard_log_dir: str | os.PathLike[str] = "",
        summary_writer: Any = None,
        summary_writer_cls: Optional[Callable[..., Any]] = None,
        warning_fn: Optional[Callable[[str], None]] = None,
    ):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.tensorboard_log_dir = Path(
            tensorboard_log_dir or (self.log_file.parent / "tensorboard")
        )
        self._warn = warning_fn or (
            lambda message: warnings.warn(message, RuntimeWarning, stacklevel=2)
        )
        self.writer = summary_writer
        if self.writer is None:
            self.writer = build_summary_writer(
                use_tensorboard=use_tensorboard,
                tensorboard_log_dir=self.tensorboard_log_dir,
                summary_writer_cls=summary_writer_cls,
                warning_fn=self._warn,
            )
        self._file = self.log_file.open("a", encoding="utf-8")
        self._tb_step = 0
        self._start_monotonic = time.monotonic()
        self._closed = False
        self._lock = threading.Lock()

    @property
    def tb_step(self) -> int:
        return self._tb_step

    def log(
        self,
        event: str,
        payload: Optional[Mapping[str, Any]] = None,
        **context: Any,
    ) -> JsonDict:
        """Write one event and return the exact strict-JSON-safe record."""

        if not str(event).strip():
            raise ValueError("event must be a non-empty string")
        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")

        with self._lock:
            if self._closed:
                raise RuntimeError("cannot log to a closed ProsEventLogger")
            record: JsonDict = dict(payload or {})
            record.update(context)
            record["event"] = str(event)
            record["tb_step"] = self._tb_step
            record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            record.setdefault("elapsed_time_sec", time.monotonic() - self._start_monotonic)
            safe_record = to_json_safe(record)
            encoded = json.dumps(safe_record, sort_keys=True, allow_nan=False)
            self._file.write(encoded + "\n")
            self._file.flush()

            if self.writer is not None:
                try:
                    write_tensorboard_scalars(
                        self.writer,
                        safe_record,
                        self._tb_step,
                        prefix=str(event),
                    )
                    self.writer.flush()
                except Exception as exc:  # Optional telemetry must not stop training.
                    self._warn(f"TensorBoard write failed ({exc}); disabling TensorBoard logging.")
                    try:
                        self.writer.close()
                    except Exception:
                        pass
                    self.writer = None

            self._tb_step += 1
            return safe_record

    log_event = log

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._file.flush()
            if self.writer is not None:
                try:
                    self.writer.flush()
                except Exception as exc:
                    self._warn(f"TensorBoard flush failed ({exc}).")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._file.flush()
            self._file.close()
            if self.writer is not None:
                try:
                    self.writer.flush()
                    self.writer.close()
                except Exception as exc:
                    self._warn(f"TensorBoard close failed ({exc}).")
            self._closed = True

    def __enter__(self) -> "ProsEventLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


ProsLogger = ProsEventLogger


__all__ = [
    "ProsEventLogger",
    "ProsLogger",
    "RewardDebugAggregator",
    "build_summary_writer",
    "clip_log_text",
    "compute_generation_perf",
    "compute_length_metrics",
    "generation_perf_metrics",
    "length_metrics",
    "make_failed_reward_sample",
    "new_reward_debug_stats",
    "record_group_decision",
    "record_reward_detail",
    "safe_div",
    "safe_divide",
    "sanitize_tag",
    "sanitize_tb_tag",
    "summarize_reward_debug",
    "to_json_safe",
    "write_tensorboard_scalars",
]
