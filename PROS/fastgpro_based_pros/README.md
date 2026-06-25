# FastGRPO-Based PROS

This package ports the PROS algorithmic pieces from `PROS/original_pros` onto the
FastGRPO training/generation stack without modifying either reference tree.

The implementation lives under the intentionally spelled target path:
`PROS/fastgpro_based_pros`.

## Migration Map

| Original PROS source | Purpose | FastGRPO-based component | Decision |
| --- | --- | --- | --- |
| `verl/experimental/dataset/tree_engine.py::TreeEngine.update_data_source` | Create augmented prompts by selecting a partial rollout from successful responses | `pros_tree.py::ProsTreeEngine.update_data_source` | Reimplemented without ray/DataProto, preserving entropy/mix window selection |
| `verl/experimental/dataset/tree_engine.py::PGTreeEngine.update_posterior/select_batch` | Track posterior success probabilities and select uncertain examples | `pros_tree.py::ProsTreeEngine` | Reimplemented with the same PG Gibbs structure; local truncated PG sampler avoids extra polyagamma dependency |
| `verl/utils/dataset/rl_dataset.py::TreeDataset.__getitem__` | Append selected partial rollout IDs to prompt IDs | `pros_trainer.py::_encode_selected_items` | Reimplemented for FastGRPO prompt tokenization |
| `verl/workers/rollout/sglang_rollout` partial rollout handling | Generate continuation after partial rollout and return full response | `pros_trainer.py::_generate/_build_training_records` | Wrapped FastGRPO `speculative_generate`; full response is partial plus new suffix |
| `ray_trainer.py` partial rollout response mask | Exclude inherited partial tokens from actor loss | `pros_trainer.py::TrainingRecord.new_token_mask` | Reimplemented; actor optimizes only newly sampled suffix tokens |
| `core_algos.py::compute_gpg_outcome_advantage` | GPG advantage option | `pros_loss.py::compute_gpg_advantages` | Reimplemented |
| `core_algos.py::compute_policy_loss` | verl clipped GRPO loss | `pros_loss.py::compute_clipped_grpo_policy_loss` | Reimplemented |
| `core_algos.py::compute_policy_loss_gpg` | GPG policy loss | `pros_loss.py::compute_gpg_policy_loss` | Reimplemented |
| `FastGRPO/grpo_speculative.py::compute_target_loss` | FastGRPO PPO-style target loss with optional reference KL | `pros_loss.py::compute_fastgrpo_policy_loss` | Reimplemented and used by default for FastGRPO-fair runs |
| `FastGRPO/helper/multitask.py` | Multi-task dataset loading, prompt rendering, and reward dispatch | `pros_trainer.py` imports `load_multitask_QAs`, `normalize_single_task_QAs`, `render_messages`, `compute_multitask_reward` | Reused |
| `FastGRPO/helper/*` | Model wrapper, speculative generation, math rewards, datasets | `pros_trainer.py` imports these helpers | Reused |

## What Was Ported

- PROS tree-augmented prompt construction.
- Entropy and mix partial-rollout selectors.
- PG-style posterior update and uncertain-example sampling.
- Partial-rollout loss masking.
- GRPO and GPG advantage computation.
- FastGRPO-compatible target loss, reward calls, speculative generation, LoRA target loading, draft model wrapper, logging, and checkpoint output.
- FastGRPO-compatible multi-task/multi-reward RLVR mode through `task_config`.
- FastGRPO-compatible `generation_backend` mode: `speculative` or target-only `target`.

## Important Deviations

- The original PROS runner is verl/ray/sglang-based. This port removes ray/DataProto and uses local Python objects so it can sit cleanly on FastGRPO.
- FastGRPO exposes much of its trainer as script-local code, not importable classes. The port reuses helper modules directly and reimplements orchestration under `pros_trainer.py`.
- The default objective is `fastgrpo` for fair FastGRPO comparison. To mirror the original `run_grpo_pros.sh` more closely, set `--objective clipped_grpo --beta 0.0 --advantage-estimator grpo`.
- The tree expects binary success for posterior updates. FastGRPO math rewards are thresholded with `tree_score_threshold` for the tree, while the continuous reward is still used for advantages.
- Exact end-to-end equivalence to the original PROS code has not been claimed; the loss/tree unit tests validate local formulas, but a full verl parity run was not possible in this lightweight environment.

## Smoke Checks

These commands do not require a model:

```bash
python3 PROS/fastgpro_based_pros/train_pros.py --dry-run true --train-draft false
python3 -m unittest discover -s PROS/fastgpro_based_pros/tests -v
```

In an environment with the FastGRPO requirements installed, the tests will execute the tensor checks instead of skipping them.

## PROS Training

Edit `configs/pros_fastgrpo_example.json`, then run:

```bash
python3 PROS/fastgpro_based_pros/train_pros.py \
  --config PROS/fastgpro_based_pros/configs/pros_fastgrpo_example.json
```

Equivalent shell wrapper:

```bash
bash PROS/fastgpro_based_pros/scripts/run_pros.sh
```

## Multi-Task / Multi-Reward PROS

The PROS trainer accepts the same task JSON used by FastGRPO:

```bash
python3 PROS/fastgpro_based_pros/train_pros.py \
  --config PROS/fastgpro_based_pros/configs/pros_multitask_fastgrpo_example.json \
  --task-config FastGRPO/configs/multitask_rlvr.example.json \
  --generation-backend speculative
```

Target-only PROS baseline with the same task/reward harness:

```bash
python3 PROS/fastgpro_based_pros/train_pros.py \
  --config PROS/fastgpro_based_pros/configs/pros_multitask_fastgrpo_example.json \
  --generation-backend target \
  --train-draft false \
  --output-dir outputs/pros_multitask_target \
  --log-file outputs/pros_multitask_target/train.jsonl \
  --saved-model-dir outputs/pros_multitask_target/target \
  --saved-draft-model-dir outputs/pros_multitask_target/draft \
  --saved-statistics-dir outputs/pros_multitask_target/statistics
```

Supported reward types are inherited from FastGRPO: `math_latex`,
`exact_match`, `contains`, `regex`, `format_only`, `code`, `zero`, and custom
reward callables through `custom_reward_func`. The built-in `code` reward is a
non-executing placeholder that checks extracted code, Python syntax,
`entry_point`, and optional expected substrings; plug sandboxed unit-test
execution in through `custom_reward_func` for real coding RLVR. Logs include
`generation_backend` and `task_metrics`.

Use these knobs for the closest original-PROS behavior:

```bash
python3 PROS/fastgpro_based_pros/train_pros.py \
  --config PROS/fastgpro_based_pros/configs/pros_fastgrpo_example.json \
  --objective clipped_grpo \
  --advantage-estimator grpo \
  --beta 0.0 \
  --tree-selector entropy \
  --tree-gibbs-sweeps 100 \
  --tree-gamma 0.99
```

## FastGRPO Baseline Comparison

Run the baseline with the same model, adapter, data option, batch size, generation
parameters, response length, reward, accumulation, and checkpoint cadence:

```bash
python3 FastGRPO/grpo_speculative.py \
  --model_dir "$MODEL_DIR" \
  --adapter_path "$DRAFT_ADAPTER_PATH" \
  --load_lora_path "" \
  --model_type qwen2 \
  --train_option simplelr_abel_level3to5 \
  --version_name fastgrpo_baseline \
  --batch_size 4 \
  --num_epochs 1 \
  --sample_num 100 \
  --accumulation_steps 2 \
  --draft_accumulation_steps 1 \
  --target_lr 1e-6 \
  --draft_lr 1e-4 \
  --is_train_draft True \
  --temperature 1.0 \
  --top_p 0.95 \
  --max_length 2048 \
  --max_training_padding_gap 256 \
  --max_training_token 3072 \
  --grpo_iteration_num 1 \
  --repeated_generate_nums 8 \
  --beta 0.0 \
  --epsilon 0.1 \
  --log_file outputs/fastgrpo_baseline/train.log \
  --saved_model_dir outputs/fastgrpo_baseline/target \
  --saved_draft_model_dir outputs/fastgrpo_baseline/draft \
  --saved_statistics_dir outputs/fastgrpo_baseline/statistics
```

For multi-task comparisons, use the same task config on both sides:

```bash
python3 FastGRPO/grpo_speculative.py \
  --model_dir "$MODEL_DIR" \
  --adapter_path "$DRAFT_ADAPTER_PATH" \
  --task_config FastGRPO/configs/multitask_rlvr.example.json \
  --generation_backend speculative \
  --model_type qwen2 \
  --version_name multitask_fastgrpo \
  --batch_size 4 \
  --num_epochs 1 \
  --repeated_generate_nums 8 \
  --is_train_draft True \
  --log_file outputs/fastgrpo_multitask/train.log \
  --saved_model_dir outputs/fastgrpo_multitask/target \
  --saved_draft_model_dir outputs/fastgrpo_multitask/draft \
  --saved_statistics_dir outputs/fastgrpo_multitask/statistics

python3 PROS/fastgpro_based_pros/train_pros.py \
  --config PROS/fastgpro_based_pros/configs/pros_multitask_fastgrpo_example.json \
  --model-dir "$MODEL_DIR" \
  --adapter-path "$DRAFT_ADAPTER_PATH" \
  --task-config FastGRPO/configs/multitask_rlvr.example.json
```

Then run PROS with matching shared settings:

```bash
python3 PROS/fastgpro_based_pros/train_pros.py \
  --config PROS/fastgpro_based_pros/configs/pros_fastgrpo_example.json \
  --model-dir "$MODEL_DIR" \
  --adapter-path "$DRAFT_ADAPTER_PATH" \
  --output-dir outputs/pros_fastgrpo
```
