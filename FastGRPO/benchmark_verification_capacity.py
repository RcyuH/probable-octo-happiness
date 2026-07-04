"""Benchmark speculative verification_capacity for a model/draft pair.

The script sweeps one or more verification_capacity values, runs
speculative_generate on representative prompts, and writes per-batch plus
aggregate throughput/memory metrics. It is intended as a practical autotuning
aid rather than a proof that one capacity is universally optimal.
"""

import argparse
import csv
import json
import math
import random
import time
from copy import deepcopy
from datetime import datetime
from itertools import cycle, islice
from pathlib import Path


FALLBACK_PROMPTS = [
    "A store sold 37 notebooks in the morning and 48 in the afternoon. "
    "It then received 25 new notebooks. How many notebooks changed hands?",
    "If a rectangle has perimeter 54 and length 17, what is its width?",
    "Write a concise Python function that returns the factorial of a non-negative integer.",
    "Explain why the sum of two even integers is always even.",
]


def load_runtime_dependencies():
    global torch
    global np
    global AutoConfig, AutoModelForCausalLM, AutoTokenizer
    global get_train_QAs, Model, DEFAULT_PROMPTS
    global load_multitask_QAs, normalize_single_task_QAs, render_messages
    global speculative_generate, get_adaptive_hyperparameters

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from helper.get_QAs import get_train_QAs
    from helper.modeling_draft import Model
    from helper.multitask import (
        DEFAULT_PROMPTS,
        load_multitask_QAs,
        normalize_single_task_QAs,
        render_messages,
    )
    from helper.specualtive_generate import speculative_generate
    from helper.speculative_hyperparameters import get_adaptive_hyperparameters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure FastGRPO target C_peak and optionally sweep verification_capacity."
    )
    parser.add_argument("--model_dir", required=True, help="Base target model directory/name.")
    parser.add_argument(
        "--adapter_path",
        default="",
        help="Draft model checkpoint path. Required for --benchmark_mode capacity/both.",
    )
    parser.add_argument(
        "--benchmark_mode",
        default="auto",
        choices=["auto", "c_peak", "capacity", "both"],
        help="auto runs c_peak only without --adapter_path, otherwise c_peak plus capacity sweep.",
    )
    parser.add_argument(
        "--target_lora_path",
        default="",
        help="Optional trained target LoRA adapter to load before benchmarking.",
    )
    parser.add_argument(
        "--capacities",
        default="64,96,128,160,192,256,320",
        help="Comma list/ranges like '64,96,128:320:32', or 'c_peak'/'auto' after C_peak measurement.",
    )
    parser.add_argument("--c_peak_b_max", type=int, default=256)
    parser.add_argument("--c_peak_step", type=int, default=8)
    parser.add_argument("--c_peak_warmup", type=int, default=10)
    parser.add_argument("--c_peak_repeat", type=int, default=30)
    parser.add_argument("--c_peak_threshold", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--repeated_generate_nums", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=3)
    parser.add_argument("--warmup_batches", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--prompt_max_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--do_sample", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--task_config", default="", help="Optional FastGRPO multitask config.")
    parser.add_argument("--task_split", default="train")
    parser.add_argument("--task_samples_per_epoch", type=int, default=None)
    parser.add_argument(
        "--train_option",
        default="",
        help="Optional legacy single-task dataset name, e.g. gsm8k_train_grpo.",
    )
    parser.add_argument(
        "--prompt_file",
        default="",
        help="Optional .txt/.jsonl prompts. JSONL supports messages, question, prompt, instruction, or text.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Inline prompt. Can be passed multiple times.",
    )
    parser.add_argument("--max_draft_token_length", type=int, default=5)
    parser.add_argument("--min_draft_token_length", type=int, default=3)
    parser.add_argument("--max_draft_k", type=int, default=8)
    parser.add_argument("--max_verification_num", type=int, default=160)
    parser.add_argument("--draft_token_length_c", type=float, default=0.75)
    parser.add_argument("--output_dir", default="benchmark_results")
    parser.add_argument("--output_prefix", default="")
    parser.add_argument(
        "--recommend_metric",
        default="generated_tokens_per_second",
        choices=["generated_tokens_per_second", "speculative_avg_emitted_tokens_per_round"],
    )
    return parser.parse_args()


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def parse_capacities(raw, c_peak=None):
    if str(raw).strip().lower() in {"auto", "c_peak"}:
        if c_peak is None:
            return None
        return [int(c_peak)]

    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [int(piece) for piece in part.split(":")]
            if len(pieces) not in {2, 3}:
                raise ValueError(f"Invalid capacity range {part!r}; use start:end[:step].")
            start, end = pieces[:2]
            step = pieces[2] if len(pieces) == 3 else 1
            if step <= 0:
                raise ValueError(f"Capacity range step must be positive: {part!r}.")
            values.extend(range(start, end + 1, step))
        else:
            values.append(int(part))
    unique = sorted(set(values))
    if not unique:
        raise ValueError("--capacities must include at least one integer.")
    return unique


def resolve_dtype(name):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_target_model_and_tokenizer(args):
    if not torch.cuda.is_available():
        raise RuntimeError("FastGRPO benchmarks currently require CUDA.")

    target_config = AutoConfig.from_pretrained(
        args.model_dir, trust_remote_code=args.trust_remote_code
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=resolve_dtype(args.torch_dtype),
        config=target_config,
        trust_remote_code=args.trust_remote_code,
    ).cuda()
    target_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return target_model, tokenizer


def maybe_load_target_lora(target_model, target_lora_path):
    if not target_lora_path:
        return target_model
    from peft import PeftModel

    target_model = PeftModel.from_pretrained(target_model, target_lora_path)
    target_model.eval()
    return target_model


def build_fastgrpo_model(args, target_model):
    if not args.adapter_path:
        raise ValueError("--adapter_path is required for capacity sweep mode.")

    draft_config = AutoConfig.from_pretrained(
        args.model_dir, trust_remote_code=args.trust_remote_code
    )
    draft_config.rope_scaling = None
    draft_config.num_hidden_layers = 1
    if getattr(draft_config, "torch_dtype", None) is None:
        draft_config.torch_dtype = target_model.dtype
    model = Model(draft_config, target_model=target_model)
    model.load_model(args.adapter_path)
    model = model.cuda()
    model.eval()
    return model


def load_examples(args):
    if args.prompt:
        return [make_generic_example(prompt, idx) for idx, prompt in enumerate(args.prompt)]

    if args.prompt_file:
        return load_prompt_file(args.prompt_file)

    if args.task_config:
        return load_multitask_QAs(
            args.task_config,
            split=args.task_split,
            samples_per_epoch=args.task_samples_per_epoch,
            seed=args.seed,
        )

    if args.train_option:
        return normalize_single_task_QAs(
            get_train_QAs(args.train_option),
            task_id=args.train_option,
            prompt_type="math",
            reward_type="math_latex",
        )

    return [make_generic_example(prompt, idx) for idx, prompt in enumerate(FALLBACK_PROMPTS)]


def make_generic_example(prompt, idx):
    defaults = DEFAULT_PROMPTS["generic"]
    return {
        "task_id": "inline",
        "question": prompt,
        "answer": "",
        "prompt_type": "generic",
        "system_prompt": defaults["system_prompt"],
        "user_prompt_template": defaults["user_prompt_template"],
        "metadata": {"source_index": idx},
    }


def load_prompt_file(path):
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    examples = []
    if prompt_path.suffix == ".jsonl":
        with prompt_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                examples.append(normalize_prompt_record(record, idx))
    else:
        with prompt_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                prompt = line.strip()
                if prompt:
                    examples.append(make_generic_example(prompt, idx))

    if not examples:
        raise ValueError(f"No prompts loaded from {prompt_path}")
    return examples


def normalize_prompt_record(record, idx):
    if "messages" in record:
        item = deepcopy(record)
        item.setdefault("task_id", "prompt_file")
        item.setdefault("answer", "")
        item.setdefault("metadata", {})
        item["metadata"].setdefault("source_index", idx)
        return item

    for key in ("question", "prompt", "instruction", "text"):
        if record.get(key):
            item = make_generic_example(str(record[key]), idx)
            item.update({k: v for k, v in record.items() if k not in item})
            return item
    raise ValueError(f"JSONL prompt record {idx} has no messages/question/prompt/instruction/text.")


def make_batches(examples, batch_size, count):
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if count <= 0:
        return []
    iterator = cycle(examples)
    return [list(islice(iterator, batch_size)) for _ in range(count)]


def tokenize_batch(tokenizer, examples, prompt_max_length):
    messages = [render_messages(example) for example in examples]
    try:
        texts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        texts = [fallback_chat_text(message) for message in messages]
    return tokenizer(
        text=texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=prompt_max_length,
        add_special_tokens=False,
    )


def fallback_chat_text(messages):
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def generation_perf_metrics(outputs, token_ids_length, wall_time):
    generated_completion_tokens = int(sum(token_ids_length))
    emitted_tokens = int(outputs.get("speculative_emitted_tokens", outputs.get("total_acc_length", 0)) or 0)
    accepted_draft_tokens = int(outputs.get("speculative_accepted_draft_tokens", 0) or 0)
    verified_draft_tokens = int(outputs.get("speculative_verified_draft_tokens", 0) or 0)
    path_budget_tokens = int(outputs.get("speculative_path_budget_tokens", 0) or 0)
    verification_rounds = int(
        outputs.get("speculative_verification_rounds", outputs.get("total_decoded_token_num", 0)) or 0
    )
    total_time = float(outputs.get("total_time_cost", wall_time) or wall_time)

    return {
        "generated_completion_tokens": generated_completion_tokens,
        "wall_time_seconds": wall_time,
        "reported_time_seconds": total_time,
        "generated_tokens_per_second": safe_div(generated_completion_tokens, wall_time),
        "speculative_verification_rounds": verification_rounds,
        "speculative_emitted_tokens": emitted_tokens,
        "speculative_accepted_draft_tokens": accepted_draft_tokens,
        "speculative_verified_draft_tokens": verified_draft_tokens,
        "speculative_path_budget_tokens": path_budget_tokens,
        "speculative_avg_emitted_tokens_per_round": safe_div(emitted_tokens, verification_rounds),
        "speculative_avg_accepted_draft_tokens_per_round": safe_div(
            accepted_draft_tokens, verification_rounds
        ),
        "speculative_path_acceptance_rate": safe_div(accepted_draft_tokens, path_budget_tokens),
        "speculative_tree_acceptance_rate": safe_div(accepted_draft_tokens, verified_draft_tokens),
        "speculative_verified_draft_tokens_per_round": safe_div(
            verified_draft_tokens, verification_rounds
        ),
        "target_time_seconds": float(outputs.get("target_time_cost", 0.0) or 0.0),
        "draft_time_seconds": float(outputs.get("draft_time_cost", 0.0) or 0.0),
        "check_time_seconds": float(outputs.get("check_time_cost", 0.0) or 0.0),
        "prefill_time_seconds": float(outputs.get("prefill_time_cost", 0.0) or 0.0),
    }


def resolve_benchmark_mode(args):
    if args.benchmark_mode == "auto":
        return "both" if args.adapter_path else "c_peak"
    return args.benchmark_mode


def measure_c_peak(
    target_model,
    tokenizer,
    b_max=256,
    step=8,
    warmup=10,
    repeat=30,
    threshold=0.95,
    device="cuda",
):
    if b_max <= 0:
        raise ValueError("--c_peak_b_max must be positive.")
    if step <= 0:
        raise ValueError("--c_peak_step must be positive.")
    if warmup < 0:
        raise ValueError("--c_peak_warmup must be non-negative.")
    if repeat <= 0:
        raise ValueError("--c_peak_repeat must be positive.")
    if not 0 < threshold <= 1:
        raise ValueError("--c_peak_threshold must be in (0, 1].")

    target_model.eval()
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id or pad_token_id for C_peak measurement.")

    rows = []
    with torch.no_grad():
        for batch_size in range(1, b_max + 1, step):
            input_ids = torch.full(
                (batch_size, 1),
                eos_id,
                dtype=torch.long,
                device=device,
            )
            try:
                torch.cuda.reset_peak_memory_stats()
                for _ in range(warmup):
                    _ = target_model(input_ids=input_ids)
                torch.cuda.synchronize()

                times = []
                for _ in range(repeat):
                    start = time.perf_counter()
                    _ = target_model(input_ids=input_ids)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)

                latency = float(np.median(times))
                throughput = batch_size / latency if latency else 0.0
                row = {
                    "batch_size": batch_size,
                    "latency_seconds": latency,
                    "throughput": throughput,
                    "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
                    "status": "ok",
                }
                rows.append(row)
                print(
                    f"C_peak probe B={batch_size}: "
                    f"latency={latency:.6f}s throughput={throughput:.4f}"
                )
            except Exception as exc:
                status = "oom" if is_oom_error(exc) else "error"
                rows.append(
                    {
                        "batch_size": batch_size,
                        "status": status,
                        "error": str(exc),
                    }
                )
                print(f"C_peak probe B={batch_size}: {status}: {exc}")
                torch.cuda.empty_cache()
                if status == "oom":
                    break

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        raise RuntimeError("C_peak measurement did not complete any successful batch size.")

    max_throughput = max(row["throughput"] for row in ok_rows)
    threshold_throughput = threshold * max_throughput
    c_peak = min(
        row["batch_size"]
        for row in ok_rows
        if row["throughput"] >= threshold_throughput
    )
    summary = {
        "c_peak": c_peak,
        "max_throughput": max_throughput,
        "threshold": threshold,
        "threshold_throughput": threshold_throughput,
        "b_max": b_max,
        "step": step,
        "warmup": warmup,
        "repeat": repeat,
    }
    return c_peak, rows, summary


def run_generation(model, tokenizer, tokenized, args, capacity, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    input_ids = tokenized["input_ids"].cuda()
    attention_mask = tokenized["attention_mask"].cuda()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = speculative_generate(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            do_sample=args.do_sample,
            max_length=args.max_length,
            repeated_generate_nums=args.repeated_generate_nums,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            verification_capacity=capacity,
            max_draft_token_length=args.max_draft_token_length,
            min_draft_token_length=args.min_draft_token_length,
            max_draft_k=args.max_draft_k,
            max_verification_num=args.max_verification_num,
            draft_token_length_c=args.draft_token_length_c,
            return_all_draft_input=False,
            statistical_time=True,
        )
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - start
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    token_ids_length = [len(item) for item in outputs["generated_token_ids"]]
    metrics = generation_perf_metrics(outputs, token_ids_length, wall_time)
    metrics["peak_memory_gb"] = peak_memory_gb
    metrics["max_sequence_length"] = int(outputs.get("max_sequence_length", 0) or 0)
    return metrics


def is_oom_error(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def summarize_capacity(capacity, batch_rows, hyperparams):
    ok_rows = [row for row in batch_rows if row["status"] == "ok"]
    if not batch_rows:
        batch_rows = [{"status": "error", "error": "no benchmark rows were recorded"}]
    base = {
        "capacity": capacity,
        "status": "ok" if len(ok_rows) == len(batch_rows) and ok_rows else batch_rows[-1]["status"],
        "batch_count": len(ok_rows),
        "verification_num": hyperparams["verification_num"],
        "draft_token_length": hyperparams["draft_token_length"],
        "draft_k": hyperparams["draft_k"],
        "draft_total_token": hyperparams["draft_total_token"],
    }
    if not ok_rows:
        base["error"] = batch_rows[-1].get("error", "")
        return base

    generated_tokens = sum(row["generated_completion_tokens"] for row in ok_rows)
    wall_time = sum(row["wall_time_seconds"] for row in ok_rows)
    emitted_tokens = sum(row["speculative_emitted_tokens"] for row in ok_rows)
    accepted_tokens = sum(row["speculative_accepted_draft_tokens"] for row in ok_rows)
    verified_tokens = sum(row["speculative_verified_draft_tokens"] for row in ok_rows)
    path_budget_tokens = sum(row["speculative_path_budget_tokens"] for row in ok_rows)
    rounds = sum(row["speculative_verification_rounds"] for row in ok_rows)
    target_time = sum(row["target_time_seconds"] for row in ok_rows)
    draft_time = sum(row["draft_time_seconds"] for row in ok_rows)
    check_time = sum(row["check_time_seconds"] for row in ok_rows)
    prefill_time = sum(row["prefill_time_seconds"] for row in ok_rows)

    base.update(
        {
            "generated_completion_tokens": generated_tokens,
            "wall_time_seconds": wall_time,
            "generated_tokens_per_second": safe_div(generated_tokens, wall_time),
            "mean_batch_wall_time_seconds": safe_div(wall_time, len(ok_rows)),
            "mean_output_tokens_per_sequence": safe_div(
                generated_tokens,
                sum(row["sequence_count"] for row in ok_rows),
            ),
            "peak_memory_gb": max(row["peak_memory_gb"] for row in ok_rows),
            "speculative_verification_rounds": rounds,
            "speculative_avg_emitted_tokens_per_round": safe_div(emitted_tokens, rounds),
            "speculative_avg_accepted_draft_tokens_per_round": safe_div(accepted_tokens, rounds),
            "speculative_path_acceptance_rate": safe_div(accepted_tokens, path_budget_tokens),
            "speculative_tree_acceptance_rate": safe_div(accepted_tokens, verified_tokens),
            "speculative_verified_draft_tokens_per_round": safe_div(verified_tokens, rounds),
            "target_time_ratio": safe_div(target_time, wall_time),
            "draft_time_ratio": safe_div(draft_time, wall_time),
            "check_time_ratio": safe_div(check_time, wall_time),
            "prefill_time_ratio": safe_div(prefill_time, wall_time),
        }
    )
    return base


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows, best_row, recommend_metric):
    print("\ncapacity,status,verification_num,draft_len,draft_k,tok/s,peak_gb,accept_path,emit/round")
    for row in summary_rows:
        print(
            "{capacity},{status},{verification_num},{draft_token_length},{draft_k},"
            "{tok_s:.4f},{mem:.3f},{accept:.4f},{emit:.4f}".format(
                capacity=row["capacity"],
                status=row["status"],
                verification_num=row.get("verification_num", ""),
                draft_token_length=row.get("draft_token_length", ""),
                draft_k=row.get("draft_k", ""),
                tok_s=float(row.get("generated_tokens_per_second", 0.0) or 0.0),
                mem=float(row.get("peak_memory_gb", 0.0) or 0.0),
                accept=float(row.get("speculative_path_acceptance_rate", 0.0) or 0.0),
                emit=float(row.get("speculative_avg_emitted_tokens_per_round", 0.0) or 0.0),
            )
        )
    if best_row:
        print(
            f"\nRecommended verification_capacity={best_row['capacity']} "
            f"by {recommend_metric}={best_row[recommend_metric]:.4f}"
        )
    else:
        print("\nNo successful capacity run; inspect the batch JSONL for errors.")


def main():
    args = parse_args()
    load_runtime_dependencies()
    benchmark_mode = resolve_benchmark_mode(args)
    run_c_peak = benchmark_mode in {"c_peak", "both"}
    run_capacity = benchmark_mode in {"capacity", "both"}

    if run_capacity and not args.adapter_path:
        raise ValueError("--adapter_path is required for --benchmark_mode capacity/both.")
    if run_capacity:
        if args.num_batches <= 0:
            raise ValueError("--num_batches must be positive.")
        if args.warmup_batches < 0:
            raise ValueError("--warmup_batches must be non-negative.")
        if args.repeated_generate_nums <= 0:
            raise ValueError("--repeated_generate_nums must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.output_prefix or datetime.now().strftime("verification_capacity_%Y%m%d_%H%M%S")
    c_peak_csv = output_dir / f"{run_id}_c_peak.csv"
    c_peak_json = output_dir / f"{run_id}_c_peak.json"
    batch_jsonl = output_dir / f"{run_id}_batches.jsonl"
    summary_csv = output_dir / f"{run_id}_summary.csv"
    summary_json = output_dir / f"{run_id}_summary.json"

    print("Loading model and tokenizer...")
    target_model, tokenizer = load_target_model_and_tokenizer(args)
    model = None
    target_for_c_peak = target_model
    defer_c_peak_until_after_capacity_model = run_c_peak and run_capacity and bool(args.target_lora_path)
    if run_c_peak and not defer_c_peak_until_after_capacity_model:
        target_for_c_peak = maybe_load_target_lora(target_model, args.target_lora_path)

    c_peak = None
    c_peak_summary = None
    if run_c_peak and not defer_c_peak_until_after_capacity_model:
        print(
            "\nMeasuring C_peak with target forward passes "
            f"(b_max={args.c_peak_b_max}, step={args.c_peak_step}, "
            f"warmup={args.c_peak_warmup}, repeat={args.c_peak_repeat})..."
        )
        c_peak, c_peak_rows, c_peak_summary = measure_c_peak(
            target_for_c_peak,
            tokenizer,
            b_max=args.c_peak_b_max,
            step=args.c_peak_step,
            warmup=args.c_peak_warmup,
            repeat=args.c_peak_repeat,
            threshold=args.c_peak_threshold,
            device="cuda",
        )
        c_peak_payload = {
            "args": vars(args),
            "benchmark_mode": benchmark_mode,
            "summary": c_peak_summary,
            "results": c_peak_rows,
        }
        write_csv(c_peak_csv, c_peak_rows)
        c_peak_json.write_text(json.dumps(c_peak_payload, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"\nC_peak={c_peak} "
            f"(max throughput={c_peak_summary['max_throughput']:.4f}, "
            f"threshold={args.c_peak_threshold:.2f})"
        )
        print(f"Wrote C_peak CSV:  {c_peak_csv}")
        print(f"Wrote C_peak JSON: {c_peak_json}")

    if not run_capacity:
        return

    model = build_fastgrpo_model(args, target_model)
    if args.target_lora_path:
        model.target_model = maybe_load_target_lora(model.target_model, args.target_lora_path)

    if defer_c_peak_until_after_capacity_model:
        print(
            "\nMeasuring C_peak with target forward passes "
            f"(b_max={args.c_peak_b_max}, step={args.c_peak_step}, "
            f"warmup={args.c_peak_warmup}, repeat={args.c_peak_repeat})..."
        )
        c_peak, c_peak_rows, c_peak_summary = measure_c_peak(
            model.target_model,
            tokenizer,
            b_max=args.c_peak_b_max,
            step=args.c_peak_step,
            warmup=args.c_peak_warmup,
            repeat=args.c_peak_repeat,
            threshold=args.c_peak_threshold,
            device="cuda",
        )
        c_peak_payload = {
            "args": vars(args),
            "benchmark_mode": benchmark_mode,
            "summary": c_peak_summary,
            "results": c_peak_rows,
        }
        write_csv(c_peak_csv, c_peak_rows)
        c_peak_json.write_text(json.dumps(c_peak_payload, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"\nC_peak={c_peak} "
            f"(max throughput={c_peak_summary['max_throughput']:.4f}, "
            f"threshold={args.c_peak_threshold:.2f})"
        )
        print(f"Wrote C_peak CSV:  {c_peak_csv}")
        print(f"Wrote C_peak JSON: {c_peak_json}")

    capacities = parse_capacities(args.capacities, c_peak=c_peak)
    if capacities is None:
        raise ValueError("--capacities auto/c_peak requires --benchmark_mode both or c_peak measurement.")

    examples = load_examples(args)
    if not examples:
        raise ValueError("No benchmark examples were loaded.")

    total_batches = args.warmup_batches + args.num_batches
    batches = make_batches(examples, args.batch_size, total_batches)
    tokenized_batches = [
        tokenize_batch(tokenizer, batch, args.prompt_max_length) for batch in batches
    ]
    for batch_index, tokenized in enumerate(tokenized_batches):
        prompt_length = tokenized["input_ids"].shape[-1]
        if prompt_length >= args.max_length:
            raise ValueError(
                f"Tokenized prompt batch {batch_index} has length {prompt_length}, "
                f"which must be smaller than --max_length {args.max_length}."
            )
    effective_bsz = args.batch_size * args.repeated_generate_nums

    print(
        f"Benchmarking {len(capacities)} capacities with active batch {effective_bsz} "
        f"({args.batch_size} prompts x {args.repeated_generate_nums} repeats)."
    )

    batch_rows = []
    summary_rows = []

    for capacity in capacities:
        print(f"\n== capacity {capacity} ==")
        capacity_rows = []
        try:
            draft_len, draft_k, draft_total = get_adaptive_hyperparameters(
                effective_bsz,
                capacity,
                args.max_draft_token_length,
                args.max_draft_k,
                args.max_verification_num,
                args.min_draft_token_length,
                args.draft_token_length_c,
            )
            verification_num = min(math.floor(capacity / effective_bsz), args.max_verification_num)
            hyperparams = {
                "verification_num": verification_num,
                "draft_token_length": draft_len,
                "draft_k": draft_k,
                "draft_total_token": draft_total,
            }
        except Exception as exc:
            row = {
                "capacity": capacity,
                "status": "invalid",
                "error": str(exc),
                "batch_index": -1,
                "is_warmup": False,
            }
            batch_rows.append(row)
            summary_rows.append(
                summarize_capacity(
                    capacity,
                    [row],
                    {
                        "verification_num": 0,
                        "draft_token_length": 0,
                        "draft_k": 0,
                        "draft_total_token": 0,
                    },
                )
            )
            print(f"invalid: {exc}")
            continue

        failed = False
        for batch_index, tokenized in enumerate(tokenized_batches):
            is_warmup = batch_index < args.warmup_batches
            try:
                metrics = run_generation(
                    model,
                    tokenizer,
                    tokenized,
                    args,
                    capacity,
                    seed=args.seed + batch_index,
                )
                row = {
                    "capacity": capacity,
                    "batch_index": batch_index - args.warmup_batches,
                    "is_warmup": is_warmup,
                    "status": "ok",
                    "prompt_count": args.batch_size,
                    "sequence_count": effective_bsz,
                    **hyperparams,
                    **metrics,
                }
                if not is_warmup:
                    capacity_rows.append(row)
                print(
                    "warmup " if is_warmup else "batch ",
                    batch_index if is_warmup else batch_index - args.warmup_batches,
                    f"tok/s={row['generated_tokens_per_second']:.4f}",
                    f"peak_gb={row['peak_memory_gb']:.3f}",
                )
            except Exception as exc:
                status = "oom" if is_oom_error(exc) else "error"
                row = {
                    "capacity": capacity,
                    "batch_index": batch_index - args.warmup_batches,
                    "is_warmup": is_warmup,
                    "status": status,
                    "prompt_count": args.batch_size,
                    "sequence_count": effective_bsz,
                    **hyperparams,
                    "error": str(exc),
                }
                print(f"{status}: {exc}")
                torch.cuda.empty_cache()
                failed = True
            batch_rows.append(row)
            if failed:
                capacity_rows.append(row)
                break

        summary_rows.append(summarize_capacity(capacity, capacity_rows, hyperparams))
        torch.cuda.empty_cache()

    successful = [row for row in summary_rows if row.get("status") == "ok"]
    best_row = max(successful, key=lambda row: row.get(args.recommend_metric, 0.0), default=None)

    write_jsonl(batch_jsonl, batch_rows)
    write_csv(summary_csv, summary_rows)
    summary_payload = {
        "recommended_capacity": best_row["capacity"] if best_row else None,
        "recommend_metric": args.recommend_metric,
        "best_row": best_row,
        "c_peak": c_peak_summary,
        "args": vars(args),
        "benchmark_mode": benchmark_mode,
        "summary": summary_rows,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    print_summary(summary_rows, best_row, args.recommend_metric)
    print(f"\nWrote batch metrics: {batch_jsonl}")
    print(f"Wrote summary CSV:   {summary_csv}")
    print(f"Wrote summary JSON:  {summary_json}")


if __name__ == "__main__":
    main()
