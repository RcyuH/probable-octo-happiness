"""Configuration helpers for the FastGRPO-based PROS port."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, get_args, get_origin, get_type_hints


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _coerce_value(raw: str, annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Optional:
        annotation = args[0]
    elif origin is not None and type(None) in args:
        annotation = next(arg for arg in args if arg is not type(None))

    if annotation is bool:
        return raw.lower() in {"1", "true", "yes", "y", "on"}
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    if annotation is Path:
        return Path(raw)
    return raw


@dataclass
class ProsConfig:
    # FastGRPO-compatible model and data inputs.
    model_dir: str = ""
    adapter_path: str = ""
    load_lora_path: str = ""
    model_type: str = "qwen2"
    train_option: str = "simplelr_abel_level3to5"
    eval_option: str = ""
    task_config: str = ""
    task_split: str = "train"
    task_samples_per_epoch: int = 0
    eval_task_config: str = ""
    eval_task_split: str = "test"

    # Output locations.
    output_dir: str = "outputs/pros_fastgrpo"
    log_file: str = "outputs/pros_fastgrpo/train.jsonl"
    saved_model_dir: str = "outputs/pros_fastgrpo/target"
    saved_draft_model_dir: str = "outputs/pros_fastgrpo/draft"
    saved_statistics_dir: str = "outputs/pros_fastgrpo/statistics"

    # Training parameters aligned with FastGRPO's grpo_speculative.py.
    num_epochs: int = 1
    batch_size: int = 4
    accumulation_steps: int = 2
    draft_accumulation_steps: int = 1
    target_lr: float = 1e-6
    draft_lr: float = 1e-4
    train_draft: bool = True
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    max_length: int = 2048
    max_training_token: int = 3072
    max_training_padding_gap: int = 256
    repeated_generate_nums: int = 8
    grpo_iteration_num: int = 1
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    generation_backend: str = "speculative"  # speculative, target
    beta: float = 0.0
    epsilon: float = 0.1

    # PROS-specific controls.
    objective: str = "fastgrpo"  # fastgrpo, clipped_grpo, gpg
    advantage_estimator: str = "grpo"  # grpo, gpg
    loss_agg_mode: str = "seq-mean-token-mean"
    drop_zero_std_groups: bool = True
    tree_selector: str = "entropy"  # entropy, mix
    tree_sampler: str = "pg"
    tree_mu0: float = 0.0
    tree_tau0: float = 1.5
    tree_sigma0: float = 0.0
    tree_delta: float = 0.1
    tree_gibbs_sweeps: int = 5
    tree_gamma: float = 0.995
    tree_min_window_tokens: int = 1000
    tree_score_threshold: float = 1.0
    tree_allow_fallback_fill: bool = True

    # Runtime and logging.
    seed: int = 42
    save_freq: int = 500
    log_freq: int = 1
    max_train_steps: int = 0
    eval_freq: int = 0
    eval_samples: int = 64
    allow_cpu: bool = False
    dry_run: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "ProsConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = _expand_env(json.load(f))
        return cls(**data)

    @classmethod
    def from_args(cls, argv: Optional[list[str]] = None) -> "ProsConfig":
        parser = argparse.ArgumentParser(description="Train PROS on top of FastGRPO infrastructure.")
        parser.add_argument("--config", type=str, default="", help="Optional JSON config file.")
        for field in fields(cls):
            name = "--" + field.name.replace("_", "-")
            parser.add_argument(name, dest=field.name, default=None)
        args = parser.parse_args(argv)

        cfg = cls.from_json(args.config) if args.config else cls()
        updates: Dict[str, Any] = {}
        type_hints = get_type_hints(cls)
        for field in fields(cls):
            raw = getattr(args, field.name)
            if raw is not None:
                updates[field.name] = _coerce_value(raw, type_hints[field.name])
        return cfg.replace(**updates)

    def replace(self, **updates: Any) -> "ProsConfig":
        data = asdict(self)
        data.update(updates)
        return type(self)(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, output_dir: str | Path | None = None) -> Path:
        out_dir = Path(output_dir or self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "pros_config.resolved.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        return path
