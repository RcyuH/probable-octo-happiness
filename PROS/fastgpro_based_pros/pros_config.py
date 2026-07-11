"""Configuration helpers for the FastGRPO-based PROS port."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, get_args, get_origin, get_type_hints


_TRUE_VALUES = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "n", "off"})
_DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _strict_bool(raw: Any) -> bool:
    """Parse explicit boolean values without silently accepting typos."""

    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    accepted = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise ValueError(f"invalid boolean value {raw!r}; expected one of: {accepted}")


def _coerce_list(raw: Any, item_annotation: Any, *, allow_comma_list: bool) -> list[Any]:
    if isinstance(raw, str):
        if not allow_comma_list:
            raise TypeError("list-valued JSON configuration fields must use a JSON array")
        values = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, list):
        values = raw
    else:
        raise TypeError(f"expected a list, got {type(raw).__name__}")
    return [_coerce_value(item, item_annotation, allow_comma_list=allow_comma_list) for item in values]


def _coerce_value(raw: Any, annotation: Any, *, allow_comma_list: bool = False) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is not None and type(None) in args:
        annotation = next(arg for arg in args if arg is not type(None))
        origin = get_origin(annotation)
        args = get_args(annotation)

    if origin is list:
        item_annotation = args[0] if args else Any
        return _coerce_list(raw, item_annotation, allow_comma_list=allow_comma_list)

    if annotation is bool:
        return _strict_bool(raw)
    if annotation is int:
        if isinstance(raw, bool):
            raise ValueError(f"expected an integer, got boolean {raw!r}")
        return int(raw)
    if annotation is float:
        if isinstance(raw, bool):
            raise ValueError(f"expected a float, got boolean {raw!r}")
        return float(raw)
    if annotation is Path:
        return Path(raw)
    if annotation is str:
        if not isinstance(raw, str):
            raise TypeError(f"expected a string, got {type(raw).__name__}")
        return raw
    return raw


def _coerce_mapping(data: Dict[str, Any], *, allow_comma_lists: bool) -> Dict[str, Any]:
    type_hints = get_type_hints(ProsConfig)
    return {
        key: _coerce_value(value, type_hints[key], allow_comma_list=allow_comma_lists)
        if key in type_hints
        else value
        for key, value in data.items()
    }


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
    lora_target_modules: list[str] = field(default_factory=lambda: list(_DEFAULT_LORA_TARGET_MODULES))
    lora_bias: str = "none"
    max_length: int = 2048
    max_training_token: int = 3072
    max_training_padding_gap: int = 256
    repeated_generate_nums: int = 8
    grpo_iteration_num: int = 1
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    generation_backend: str = "speculative"  # speculative, target
    verification_capacity: int = 160
    max_draft_token_length: int = 5
    min_draft_token_length: int = 3
    max_draft_k: int = 8
    max_verification_num: int = 160
    draft_token_length_c: float = 0.75
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
    use_tensorboard: bool = True
    tensorboard_log_dir: str = ""
    reward_debug_sample_count: int = 5
    reward_debug_sample_chars: int = 800

    def __post_init__(self) -> None:
        # Direct construction and replace() should be as strict as JSON/CLI loading.
        type_hints = get_type_hints(type(self))
        for config_field in fields(self):
            annotation = type_hints[config_field.name]
            if annotation is bool:
                setattr(self, config_field.name, _strict_bool(getattr(self, config_field.name)))

        modules = self.lora_target_modules
        if not isinstance(modules, list):
            raise TypeError("lora_target_modules must be a list of module names")
        if not all(isinstance(module, str) for module in modules):
            raise TypeError("lora_target_modules entries must be strings")
        self.lora_target_modules = [module.strip() for module in modules if module.strip()]
        self.validate()

    def validate(self) -> "ProsConfig":
        """Validate all controls needed before trainer/model initialization."""

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.repeated_generate_nums <= 0:
            raise ValueError("repeated_generate_nums must be positive")
        if self.verification_capacity <= 0:
            raise ValueError("verification_capacity must be positive")
        if self.max_draft_k <= 0:
            raise ValueError("max_draft_k must be positive")
        if self.min_draft_token_length <= 0 or self.max_draft_token_length <= 0:
            raise ValueError("min_draft_token_length and max_draft_token_length must be positive")
        if self.min_draft_token_length > self.max_draft_token_length:
            raise ValueError("min_draft_token_length must be <= max_draft_token_length")
        if self.max_verification_num <= 1:
            raise ValueError("max_verification_num must be greater than 1")
        if self.draft_token_length_c <= 0:
            raise ValueError("draft_token_length_c must be positive")
        if self.generation_backend not in {"speculative", "target"}:
            raise ValueError("generation_backend must be 'speculative' or 'target'")
        if self.generation_backend == "speculative":
            minimum_capacity = 2 * self.batch_size * self.repeated_generate_nums
            if self.verification_capacity < minimum_capacity:
                raise ValueError(
                    "verification_capacity is too small for speculative generation: "
                    f"got {self.verification_capacity}, but batch_size {self.batch_size} * "
                    f"repeated_generate_nums {self.repeated_generate_nums} requires at least "
                    f"{minimum_capacity} verification slots"
                )

        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not self.lora_target_modules:
            raise ValueError("lora_target_modules must contain at least one module name")
        if self.lora_bias not in {"none", "all", "lora_only"}:
            raise ValueError("lora_bias must be one of: none, all, lora_only")
        if self.load_lora_path and not self.use_lora:
            raise ValueError("load_lora_path requires use_lora=true")
        if (
            self.objective == "fastgrpo"
            and self.beta > 0
            and (not self.use_lora or self.lora_bias != "none")
        ):
            raise ValueError(
                "beta > 0 requires use_lora=true and lora_bias='none' so the "
                "adapter-disabled reference policy remains frozen"
            )

        if self.reward_debug_sample_count < 0:
            raise ValueError("reward_debug_sample_count must be non-negative")
        if self.reward_debug_sample_chars <= 0:
            raise ValueError("reward_debug_sample_chars must be positive")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "ProsConfig":
        expanded_path = Path(os.path.expandvars(os.fspath(path))).expanduser()
        with open(expanded_path, "r", encoding="utf-8") as f:
            data = _expand_env(json.load(f))
        if not isinstance(data, dict):
            raise TypeError("PROS config JSON must contain an object at the top level")
        return cls(**_coerce_mapping(data, allow_comma_lists=False))

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
                updates[field.name] = _coerce_value(
                    raw,
                    type_hints[field.name],
                    allow_comma_list=True,
                )
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
