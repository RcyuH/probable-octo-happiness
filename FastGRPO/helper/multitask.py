"""Utilities for multi-task RLVR data loading and reward dispatch."""

import csv
import importlib
import json
import os
import random
from functools import lru_cache
from pathlib import Path

import datasets
import pandas as pd

from helper.rewards import compute_reward_from_example


MATH_SYSTEM_PROMPT = "You are a math problem assistant."
MATH_USER_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}"""

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
        "system_prompt": "You are a careful programming assistant.",
        "user_prompt_template": "{instruction}",
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
    "instruction",
    "input",
)
DEFAULT_ANSWER_FIELDS = (
    "reward_model.ground_truth",
    "ground_truth",
    "answer",
    "solution",
    "label",
)


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

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_template.format_map({"instruction": instruction})},
    ]


def compute_multitask_reward(completion, example):
    """Compute reward from a built-in reward type or a custom reward callable."""
    custom_reward_func = example.get("custom_reward_func")
    if custom_reward_func:
        reward_func = _load_callable(custom_reward_func)
        return float(reward_func(completion=completion, example=example))
    return compute_reward_from_example(completion, example)


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
    normalized = []

    for idx, record in enumerate(records):
        prompt = _pick_value(record, task.get("prompt_field"), DEFAULT_PROMPT_FIELDS)
        answer = _pick_value(record, task.get("answer_field"), DEFAULT_ANSWER_FIELDS)
        answer = _transform_answer(_stringify_field(answer), task)
        metadata = _extract_metadata(record, task)

        item = {
            "task_id": task_id,
            "question": _stringify_field(prompt),
            "answer": answer,
            "reward_type": task.get("reward_type", "math_latex"),
            "prompt_type": prompt_type,
            "system_prompt": task.get("system_prompt", prompt_defaults["system_prompt"]),
            "user_prompt_template": task.get("user_prompt_template", prompt_defaults["user_prompt_template"]),
            "format_weight": task.get("format_weight", 0.2),
            "task_weight": float(task.get("weight", 1.0)),
            "metadata": metadata,
        }

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
            return pd.read_parquet(path_obj).to_dict(orient="records")
        if suffix in (".json", ".jsonl"):
            return _read_json_records(path_obj)
        if suffix in (".csv", ".tsv"):
            delimiter = "\t" if suffix == ".tsv" else ","
            with open(path_obj, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f, delimiter=delimiter))
        raise ValueError(f"Unsupported dataset file type: {path}")

    dataset_split = task.get("split", split)
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
