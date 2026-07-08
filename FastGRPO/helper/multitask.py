"""Utilities for multi-task RLVR data loading and reward dispatch."""

import csv
import importlib
import json
import math
import os
import random
from functools import lru_cache
from pathlib import Path

try:
    import datasets
except ModuleNotFoundError:
    datasets = None
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from helper.rewards import compute_reward_debug_from_example, compute_reward_from_example


MATH_SYSTEM_PROMPT = "You are a math problem assistant."
MATH_USER_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}"""
CODE_SYSTEM_PROMPT = "You are a careful programming assistant."
CODE_USER_PROMPT = """Solve the following coding problem in {language}.
Return only the final solution code unless the problem explicitly asks for an explanation.

Problem:
{instruction}
"""

DEFAULT_PROMPTS = {
    "math": {
        "system_prompt": MATH_SYSTEM_PROMPT,
        "user_prompt_template": MATH_USER_PROMPT,
    },
    "qa": {
        "system_prompt": "You are a helpful assistant.",
        "user_prompt_template": "{instruction}",
    },
    "code": {
        "system_prompt": CODE_SYSTEM_PROMPT,
        "user_prompt_template": CODE_USER_PROMPT,
    },
    "generic": {
        "system_prompt": "You are a helpful assistant.",
        "user_prompt_template": "{instruction}",
    },
}

DEFAULT_PROMPT_FIELDS = (
    "prompt.0.content",
    "prompt",
    "question",
    "problem",
    "description",
    "task",
    "instruction",
    "input",
)
DEFAULT_ANSWER_FIELDS = (
    "reward_model.ground_truth",
    "ground_truth",
    "answer",
    "solution",
    "canonical_solution",
    "reference_solution",
    "label",
)
DEFAULT_LANGUAGE_FIELDS = ("language", "lang", "programming_language")
DEFAULT_ENTRY_POINT_FIELDS = (
    "entry_point",
    "function_name",
    "metadata.entry_point",
    "metadata.function_name",
)
DEFAULT_TEST_FIELDS = (
    "tests",
    "test",
    "unit_tests",
    "test_cases",
    "metadata.tests",
    "metadata.test_cases",
)
DEFAULT_TEST_TYPE_FIELDS = ("test_type", "code_test_type", "metadata.test_type")
DEFAULT_TIMEOUT_FIELDS = (
    "timeout_seconds",
    "code_timeout_seconds",
    "timeout",
    "metadata.timeout_seconds",
    "metadata.code_timeout_seconds",
)
DEFAULT_EXPECTED_SUBSTRING_FIELDS = (
    "expected_substrings",
    "required_substrings",
    "metadata.expected_substrings",
)
DEFAULT_STARTER_CODE_FIELDS = ("starter_code", "boilerplate", "metadata.starter_code")


def load_multitask_QAs(config_path, split="train", samples_per_epoch=None, seed=42):
    """Load and mix task datasets from a JSON config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tasks = config["tasks"] if isinstance(config, dict) else config
    if not tasks:
        raise ValueError(f"No tasks found in config: {config_path}")

    samples_by_task = []
    for task in tasks:
        task_samples = _load_task_samples(task, split=split)
        if not task_samples:
            continue
        samples_by_task.append((task, task_samples))

    if not samples_by_task:
        raise ValueError(f"No samples loaded from config: {config_path}")

    config_total = None
    if isinstance(config, dict):
        config_total = config.get("samples_per_epoch")
    total_samples = samples_per_epoch or config_total
    return _mix_samples(samples_by_task, samples_per_epoch=total_samples, seed=seed)


def normalize_single_task_QAs(samples, task_id, prompt_type="math", reward_type="math_latex"):
    """Attach task metadata to legacy single-task Q/A records."""
    normalized = []
    prompt_defaults = DEFAULT_PROMPTS.get(prompt_type, DEFAULT_PROMPTS["generic"])
    for idx, sample in enumerate(samples):
        item = dict(sample)
        item.setdefault("task_id", task_id)
        item.setdefault("prompt_type", prompt_type)
        item.setdefault("reward_type", reward_type)
        item.setdefault("system_prompt", prompt_defaults["system_prompt"])
        item.setdefault("user_prompt_template", prompt_defaults["user_prompt_template"])
        item.setdefault("metadata", {})
        item["metadata"].setdefault("source_index", idx)
        normalized.append(item)
    return normalized


def render_messages(example):
    """Render an example to chat messages for the tokenizer chat template."""
    if example.get("messages"):
        return example["messages"]

    prompt_type = example.get("prompt_type", "generic")
    defaults = DEFAULT_PROMPTS.get(prompt_type, DEFAULT_PROMPTS["generic"])
    system_prompt = example.get("system_prompt") or defaults["system_prompt"]
    user_prompt_template = example.get("user_prompt_template") or defaults["user_prompt_template"]
    instruction = example.get("question") or example.get("prompt") or example.get("instruction") or ""
    format_values = _SafeFormatDict(example)
    format_values["instruction"] = instruction
    if not format_values.get("language"):
        format_values["language"] = "python"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_template.format_map(format_values)},
    ]


def compute_multitask_reward(completion, example):
    """Compute reward from a built-in reward type or a custom reward callable."""
    custom_reward_func = example.get("custom_reward_func")
    if custom_reward_func:
        reward_func = _load_callable(custom_reward_func)
        return float(reward_func(completion=completion, example=example))
    return compute_reward_from_example(completion, example)


def compute_multitask_reward_debug(completion, example):
    """Compute reward and return a diagnostics dict for logging/debugging."""
    custom_reward_func = example.get("custom_reward_func")
    if custom_reward_func:
        reward_func = _load_callable(custom_reward_func)
        reward = float(reward_func(completion=completion, example=example))
        return {
            "reward": reward,
            "reward_type": str(example.get("reward_type", "custom")),
            "custom_reward": True,
            "passed": reward > 0,
            "error_type": "custom_reward_failed" if reward <= 0 else "none",
        }
    return compute_reward_debug_from_example(completion, example)


def _load_task_samples(task, split="train"):
    task_id = task.get("id") or task.get("task_id")
    if not task_id:
        raise ValueError(f"Task is missing id/task_id: {task}")

    records = _read_records(task, split=split)
    max_samples = task.get("max_samples")
    if max_samples is not None:
        records = records[: int(max_samples)]

    prompt_type = task.get("prompt_type", "generic")
    prompt_defaults = DEFAULT_PROMPTS.get(prompt_type, DEFAULT_PROMPTS["generic"])
    default_reward_type = _default_reward_type(prompt_type)
    normalized = []

    for idx, record in enumerate(records):
        prompt = _pick_value(record, task.get("prompt_field"), DEFAULT_PROMPT_FIELDS)
        answer = _pick_value(record, task.get("answer_field"), DEFAULT_ANSWER_FIELDS)
        answer = _transform_answer(_stringify_field(answer), task)
        metadata = _extract_metadata(record, task)
        language = _pick_value(record, task.get("language_field"), DEFAULT_LANGUAGE_FIELDS)
        language = task.get("language", language)
        entry_point = _pick_value(record, task.get("entry_point_field"), DEFAULT_ENTRY_POINT_FIELDS)
        entry_point = task.get("entry_point", entry_point)
        tests = _pick_value(record, task.get("tests_field"), DEFAULT_TEST_FIELDS)
        test_type = _pick_value(record, task.get("test_type_field"), DEFAULT_TEST_TYPE_FIELDS)
        test_type = task.get("test_type", test_type)
        timeout_seconds = _pick_value(record, task.get("timeout_field"), DEFAULT_TIMEOUT_FIELDS)
        timeout_seconds = task.get("timeout_seconds", task.get("code_timeout_seconds", timeout_seconds))
        expected_substrings = _pick_value(
            record,
            task.get("expected_substrings_field"),
            DEFAULT_EXPECTED_SUBSTRING_FIELDS,
        )
        starter_code = _pick_value(record, task.get("starter_code_field"), DEFAULT_STARTER_CODE_FIELDS)

        item = {
            "task_id": task_id,
            "question": _stringify_field(prompt),
            "answer": answer,
            "reward_type": task.get("reward_type", default_reward_type),
            "prompt_type": prompt_type,
            "system_prompt": task.get("system_prompt", prompt_defaults["system_prompt"]),
            "user_prompt_template": task.get("user_prompt_template", prompt_defaults["user_prompt_template"]),
            "format_weight": task.get("format_weight", 0.2),
            "task_weight": float(task.get("weight", 1.0)),
            "_task_weight_explicit": "weight" in task,
            "metadata": metadata,
        }
        if language is not None:
            item["language"] = _stringify_field(language)
        if entry_point is not None:
            item["entry_point"] = _stringify_field(entry_point)
        if tests is not None:
            item["tests"] = tests
        if test_type is not None:
            item["test_type"] = _stringify_field(test_type)
        if timeout_seconds is not None:
            item["timeout_seconds"] = float(timeout_seconds)
        if expected_substrings is not None:
            item["expected_substrings"] = _normalize_list_field(expected_substrings)
        if starter_code is not None:
            item["starter_code"] = _stringify_field(starter_code)

        if task.get("custom_reward_func"):
            item["custom_reward_func"] = task["custom_reward_func"]
        if task.get("pattern"):
            item["pattern"] = task["pattern"]
        if task.get("messages_field"):
            messages = _get_nested(record, task["messages_field"])
            if messages:
                item["messages"] = messages

        metadata.setdefault("source_index", idx)
        normalized.append(item)

    return normalized


class _SafeFormatDict(dict):
    def __init__(self, example):
        super().__init__(example)
        metadata = example.get("metadata") or {}
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                self.setdefault(key, value)

    def __missing__(self, key):
        return ""


def _default_reward_type(prompt_type):
    if prompt_type == "code":
        return "code"
    return "math_latex"


def _normalize_list_field(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [_stringify_field(item) for item in value if item is not None]
    return [_stringify_field(value)]


def _read_records(task, split="train"):
    if "records" in task:
        return list(task["records"])

    path = task.get("path") or task.get("dataset")
    if not path:
        raise ValueError(f"Task is missing path/dataset/records: {task}")

    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if path_obj.exists() and path_obj.is_file():
        if suffix == ".parquet":
            if pd is None:
                raise ModuleNotFoundError("Reading parquet task data requires pandas.")
            return pd.read_parquet(path_obj).to_dict(orient="records")
        if suffix in (".json", ".jsonl"):
            return _read_json_records(path_obj)
        if suffix in (".csv", ".tsv"):
            delimiter = "\t" if suffix == ".tsv" else ","
            with open(path_obj, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f, delimiter=delimiter))
        raise ValueError(f"Unsupported dataset file type: {path}")

    dataset_split = task.get("split", split)
    if datasets is None:
        raise ModuleNotFoundError("Loading Hugging Face datasets requires the datasets package.")
    dataset = datasets.load_dataset(path, split=dataset_split)
    return list(dataset)


def _read_json_records(path):
    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"JSON dataset must be a list or contain data/records/examples: {path}")


def _mix_samples(samples_by_task, samples_per_epoch=None, seed=42):
    rng = random.Random(seed)
    has_explicit_weight = any("weight" in task for task, _ in samples_by_task)

    if not has_explicit_weight:
        mixed = []
        for _, samples in samples_by_task:
            mixed.extend(samples)
        rng.shuffle(mixed)
        return mixed

    weights = [float(task.get("weight", 1.0)) for task, _ in samples_by_task]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Task weights must sum to a positive value.")

    total = int(samples_per_epoch or sum(len(samples) for _, samples in samples_by_task))
    counts = _weighted_counts(total, weights)

    mixed = []
    for count, (_, samples) in zip(counts, samples_by_task):
        if count <= 0:
            continue
        if count <= len(samples):
            mixed.extend(rng.sample(samples, count))
        else:
            mixed.extend(samples)
            mixed.extend(rng.choice(samples) for _ in range(count - len(samples)))

    rng.shuffle(mixed)
    return mixed


def has_explicit_task_weights(samples):
    task_ids = {sample.get("task_id", "default") for sample in samples}
    return len(task_ids) > 1 and any(sample.get("_task_weight_explicit") for sample in samples)


class TaskWeightedBatchSampler:
    """Yield batches whose task composition follows per-task weights."""

    def __init__(self, samples, batch_size, seed=42, drop_last=False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.indices_by_task = {}
        self.weights_by_task = {}

        for idx, sample in enumerate(samples):
            task_id = sample.get("task_id", "default")
            self.indices_by_task.setdefault(task_id, []).append(idx)
            self.weights_by_task.setdefault(task_id, float(sample.get("task_weight", 1.0)))

        self.task_ids = [task_id for task_id, indices in self.indices_by_task.items() if indices]
        if not self.task_ids:
            raise ValueError("TaskWeightedBatchSampler requires at least one sample.")

        self.weights = [max(0.0, self.weights_by_task.get(task_id, 1.0)) for task_id in self.task_ids]
        if sum(self.weights) <= 0:
            raise ValueError("Task weights must sum to a positive value.")

        self.total_samples = len(samples)
        if self.drop_last:
            self.total_samples = (self.total_samples // self.batch_size) * self.batch_size

    def __len__(self):
        if self.total_samples == 0:
            return 0
        if self.drop_last:
            return self.total_samples // self.batch_size
        return math.ceil(self.total_samples / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools = {}
        cursors = {}
        for task_id in self.task_ids:
            indices = list(self.indices_by_task[task_id])
            rng.shuffle(indices)
            pools[task_id] = indices
            cursors[task_id] = 0

        previous_counts = [0] * len(self.task_ids)
        emitted = 0
        while emitted < self.total_samples:
            current_batch_size = min(self.batch_size, self.total_samples - emitted)
            cumulative_counts = _weighted_counts(emitted + current_batch_size, self.weights)
            batch_counts = [
                cumulative_count - previous_count
                for cumulative_count, previous_count in zip(cumulative_counts, previous_counts)
            ]
            previous_counts = cumulative_counts

            batch = []
            for task_id, count in zip(self.task_ids, batch_counts):
                if count <= 0:
                    continue
                batch.extend(self._draw_indices(task_id, count, pools, cursors, rng))

            rng.shuffle(batch)
            emitted += len(batch)
            if batch:
                yield batch

    def _draw_indices(self, task_id, count, pools, cursors, rng):
        drawn = []
        while len(drawn) < count:
            pool = pools[task_id]
            cursor = cursors[task_id]
            remaining = len(pool) - cursor
            need = count - len(drawn)

            if remaining >= need:
                drawn.extend(pool[cursor:cursor + need])
                cursors[task_id] = cursor + need
            else:
                if remaining > 0:
                    drawn.extend(pool[cursor:])
                rng.shuffle(pool)
                cursors[task_id] = 0

        return drawn

    def batch_task_counts(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        return dict(zip(self.task_ids, _weighted_counts(batch_size, self.weights)))


def _weighted_counts(total, weights):
    total_weight = sum(weights)
    raw_counts = [total * weight / total_weight for weight in weights]
    counts = [int(count) for count in raw_counts]
    remaining = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda idx: raw_counts[idx] - counts[idx], reverse=True)
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts


def _pick_value(record, configured_field, default_fields):
    fields = configured_field if configured_field is not None else default_fields
    if isinstance(fields, str):
        fields = (fields,)

    for field in fields:
        value = _get_nested(record, field)
        if value is not None:
            return value
    return None


def _get_nested(record, field):
    if field is None:
        return None
    current = record
    for part in str(field).split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)) and part.isdigit():
            idx = int(part)
            if idx >= len(current):
                return None
            current = current[idx]
        else:
            return None
    return current


def _stringify_field(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if all(isinstance(item, dict) and "content" in item for item in value):
            return "\n".join(str(item["content"]) for item in value)
        parts = [_stringify_field(item) for item in value]
        return "\n".join(str(item) for item in parts if item is not None)
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        if "ground_truth" in value:
            return str(value["ground_truth"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _transform_answer(answer, task):
    if answer is None:
        return None

    extraction = task.get("answer_extraction")
    if extraction in ("gsm8k_hash", "last_hash") and "####" in answer:
        answer = answer.split("####")[-1].strip()

    answer_template = task.get("answer_template")
    if answer_template:
        answer = answer_template.format_map({"answer": answer})

    return answer


def _extract_metadata(record, task):
    metadata = {}
    metadata_field = task.get("metadata_field")
    if metadata_field:
        value = _get_nested(record, metadata_field)
        if isinstance(value, dict):
            metadata.update(value)

    for source_field in task.get("metadata_fields", []):
        value = _get_nested(record, source_field)
        if value is not None:
            metadata[source_field] = value

    metadata["source_task"] = task.get("id") or task.get("task_id")
    return metadata


@lru_cache(maxsize=None)
def _load_callable(spec):
    if ":" in spec:
        module_name, func_name = spec.split(":", 1)
    else:
        module_name, func_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)
