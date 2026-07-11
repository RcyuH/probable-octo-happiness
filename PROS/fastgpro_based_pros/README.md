# FastGRPO-Based PROS

This package ports PROS (Prefix Reuse for On-policy Sampling) onto the local
FastGRPO training and generation stack. It reuses promising prefixes from
successful historical rollouts as augmented prompts, then prioritizes uncertain
root and child nodes with the PROS hierarchical Bayesian sampler.

The implementation lives under the intentionally spelled path
`PROS/fastgpro_based_pros`. The reference trees under `PROS/original_pros` and
`FastGRPO` remain separate.

## Preserved PROS Semantics

- A child prompt appends its inherited partial rollout to the original prompt.
- Reward verification receives the full response: inherited partial rollout plus
  the newly generated suffix.
- `new_token_mask` excludes the original prompt and inherited prefix, so actor
  loss optimizes only newly sampled suffix tokens.
- A child always inherits the task ID of its original root ancestor.
- Continuous rewards are used unchanged by GRPO/GPG advantage and policy-loss
  computation. Only tree success/failure is binarized with
  `tree_score_threshold`.
- Entropy and mix prefix selection, Polya-Gamma posterior sampling, posterior
  uncertainty ranking, and original-ancestor diversity are retained.
- The `fastgrpo`, `clipped_grpo`, and `gpg` policy objectives remain available.

The original implementation is verl/ray/sglang-based. This port uses local
Python objects and FastGRPO helpers instead of ray `DataProto` orchestration, so
exact end-to-end equivalence to the original runtime is not claimed.

The default `tree_selector=entropy` uses target-model token entropy directly.
The local `mix` selector preserves the original group-wide 80th-percentile
filtering rule, but this lightweight port has no separate critic and therefore
uses target token log-probability as its value proxy. Use `entropy` when a
critic-free comparison closest to the supported default is desired; adding a
learned critic is outside this package's scope.

## Quick Start

The shell wrappers resolve the repository root themselves and can be launched
from any working directory. Set the model and draft-weight locations first:

```bash
export MODEL_DIR=/path/to/target-model
export DRAFT_ADAPTER_PATH=/path/to/draft-weights

bash PROS/fastgpro_based_pros/scripts/run_pros.sh
```

The same run without the wrapper is:

```bash
python3 PROS/fastgpro_based_pros/train_pros.py \
  --config PROS/fastgpro_based_pros/configs/pros_fastgrpo_example.json
```

Command-line options use kebab case and override JSON values. Environment
variables in JSON strings are expanded before validation. The resolved
configuration is written to `pros_config.resolved.json` under `output_dir`.

Run model-free checks with:

```bash
bash PROS/fastgpro_based_pros/scripts/smoke_test.sh
```

## Portable Multi-Task Math + Code Example

The bundled multi-task example is self-contained: its task file uses inline
records and does not refer to a machine-specific path or an unbundled dataset.

```bash
python3 PROS/fastgpro_based_pros/train_pros.py \
  --config PROS/fastgpro_based_pros/configs/pros_multitask_fastgrpo_example.json
```

The two relevant files are:

- `configs/pros_multitask_fastgrpo_example.json`: trainer/model settings.
- `configs/pros_multitask_math_code_tasks.example.json`: inline math and Python
  tasks, rewards, tests, and task weights.

For real workloads, a task may use inline `records`, a Hugging Face dataset name,
or a local Parquet, JSON/JSONL, CSV, or TSV file. Field selectors may be a dotted
nested path such as `metadata.tests` or a list of fallback paths. The shared
loader also supports `messages_field`, metadata mappings, deterministic seeding,
train/evaluation splits, and `math`, `qa`, `code`, and `generic` prompt types.

A minimal task entry looks like:

```json
{
  "id": "python_inline",
  "weight": 0.5,
  "prompt_type": "code",
  "reward_type": "code_unit_test",
  "prompt_field": "question",
  "language": "python",
  "entry_point_field": "entry_point",
  "tests_field": "tests",
  "timeout_seconds": 2.0,
  "records": [
    {
      "question": "Implement add(a, b).",
      "entry_point": "add",
      "tests": "assert add(2, 3) == 5"
    }
  ]
}
```

Custom reward callables may use either `package.module:function` or
`package.module.function` in `custom_reward_func`.

## Task-Aware PROS Tree Sampling

Explicit task weights affect both root-pool construction and every batch chosen
by `ProsTreeEngine`; PROS does not attach a conventional DataLoader sampler.

For a weighted multi-task pool, the tree computes deterministic
largest-remainder quotas. Weights `5:3` with `batch_size=8`, for example, request
five items from the first task and three from the second. Within each quota,
nodes remain ordered by posterior uncertainty (estimated pass rate nearest
`0.5`) and the original PROS recency/diversity rules remain active.

Selection has the following guarantees:

- A child is assigned through its original ancestor's task ID.
- Two selected nodes never share the same original ancestor.
- Zero-weight tasks are excluded; all-zero/non-positive weights are rejected.
- If a task lacks eligible candidates, fallback first relaxes recency within
  that task, then deterministically backfills from positive-weight tasks while
  preserving ancestor diversity.
- Quotas, selected counts, fallback counts, and true pre-backfill shortfalls are
  logged under `sampler/task_*`.
- With no explicit weights, or with a single task, the legacy PROS selector path
  is used unchanged.

The root Q/A pool remains fixed during training because tree nodes store stable
ancestor indices into it.

## Finite Training Lifecycle and Rollout Eligibility

Each epoch performs
`max(1, ceil(number_of_root_examples / batch_size))` generation attempts. The
`generation_attempt` counter advances even when no actor update is possible, so
empty or all-skipped batches cannot create an infinite loop. `max_train_steps`
is an optional upper bound on actor/global training steps; `step` advances only
when an actor update occurs.

Verified completions are split into two views:

- `all_records` contains every verified rollout. Reward diagnostics, per-task
  statistics, posterior updates, and tree prefix construction consume this
  complete view, including all-correct and all-wrong groups.
- `actor_records` contains only groups eligible for advantage/loss computation.
  With `drop_zero_std_groups=true`, zero-variance groups are excluded from actor
  optimization; groups whose completions are all empty are also excluded.

This distinction prevents zero-variance filtering from making the PROS tree
blind to observed outcomes. If `actor_records` is empty, the trainer still
updates/selects the tree and writes a `generation` event, but it does not write a
`train` event.

## Reward System

Each completion is evaluated exactly once through the diagnostic reward
dispatcher. The scalar in its returned `reward` field drives training, while the
same result supplies logging diagnostics.

Canonical reward types are:

| Reward | Behavior |
| --- | --- |
| `math_latex` | Math verification plus optional format reward. If no `\boxed{}` is present, an `<answer>...</answer>` value can be converted for parsing. |
| `exact_match` | Normalized exact comparison with the reference answer. |
| `contains` | Checks whether the normalized reference answer occurs in the completion. |
| `regex` | Matches the task's configured `pattern`. |
| `format_only` | Applies only the reasoning/answer formatting check. |
| `code` | Static, non-executing code checks: non-empty extraction, Python syntax when applicable, optional entry point, and optional expected substrings. |
| `code_unit_test` | Executes generated Python against configured tests in a local subprocess with a timeout. |
| `zero` | Always returns `0.0`. |
| custom | Calls `custom_reward_func(completion=..., example=...)`. |

Aliases accepted by the shared FastGRPO dispatcher continue to work, but the
canonical names above are recommended in new task files.

### Code Verifier Security

`code_unit_test` supports fenced-code extraction, starter code, plain assertions,
plain `test_*` functions, `unittest.TestCase`, and stdin/stdout test records. It
classifies outcomes such as empty completion/tests, syntax, runtime, import,
assertion, wrong answer, timeout, and unsupported language.

The verifier is a bounded **local subprocess runner, not a hardened sandbox**.
It executes model-generated Python on the training machine. Use it only with
datasets and generated code you are willing to execute locally. For untrusted
code, configure a custom reward that delegates execution to a real external
sandbox. The static `code` reward never launches a subprocess.

For partial-rollout nodes, both static and executing code rewards receive the
full inherited-prefix-plus-suffix completion, not only the suffix.

## Speculative-Decoding Controls

The six controls below are present in JSON, kebab-case CLI flags, the resolved
config, and all speculative generation calls during both training and evaluation.

| JSON key | CLI flag | Default |
| --- | --- | ---: |
| `verification_capacity` | `--verification-capacity` | `160` |
| `max_draft_token_length` | `--max-draft-token-length` | `5` |
| `min_draft_token_length` | `--min-draft-token-length` | `3` |
| `max_draft_k` | `--max-draft-k` | `8` |
| `max_verification_num` | `--max-verification-num` | `160` |
| `draft_token_length_c` | `--draft-token-length-c` | `0.75` |

All sizes/constants must be positive, `min_draft_token_length` must not exceed
`max_draft_token_length`, and `max_verification_num` must be greater than one.
For `generation_backend=speculative`, validation also requires:

```text
verification_capacity >= 2 * batch_size * repeated_generate_nums
```

The target-only backend does not use these controls and bypasses only this
minimum-capacity relationship; the values must still be individually valid.

## LoRA Controls

| JSON key | CLI flag | Default |
| --- | --- | --- |
| `use_lora` | `--use-lora` | `true` |
| `load_lora_path` | `--load-lora-path` | empty |
| `lora_r` | `--lora-r` | `64` |
| `lora_alpha` | `--lora-alpha` | `32` |
| `lora_dropout` | `--lora-dropout` | `0.0` |
| `lora_target_modules` | `--lora-target-modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| `lora_bias` | `--lora-bias` | `none` |

JSON requires `lora_target_modules` to be an array; the CLI accepts a
comma-separated string. `lora_bias` accepts `none`, `all`, or `lora_only`.
Rank/alpha must be positive, dropout must be in `[0, 1)`, and the module list
must not be empty. `load_lora_path` requires `use_lora=true`.

FastGRPO-objective KL regularization with `beta>0` requires `use_lora=true` and
`lora_bias=none`. In that configuration, disabling the adapter exposes the
frozen base model used as the reference policy. Fully trainable targets and
trainable LoRA bias modes do not provide a frozen reference copy, so the
configuration validator rejects that combination instead of logging a
misleading KL objective. The `clipped_grpo` and `gpg` objectives do not consume
`beta` or reference log probabilities.

When LoRA is disabled, PEFT is not imported or wrapped, the target model remains
fully trainable, and the optimizer still filters to parameters with
`requires_grad=true`. `adapter_path` is the draft-model weight path and is
independent of `load_lora_path`. Target and draft models enter evaluation mode
before generation; actor and draft training switch their respective models back
to training mode only for optimization.

## Target-Only vs. Speculative Generation

| Property | `speculative` | `target` |
| --- | --- | --- |
| Generator | FastGRPO speculative generator | Target model's `generate()` |
| Draft model training | Optional with `train_draft=true` | Disabled |
| Six speculative controls | Applied to train and eval | Ignored |
| Capacity relationship | Enforced | Bypassed |
| Reward, PROS tree, actor loss, event schema | Shared | Shared |
| Speculative metric fields | Populated when provided by generator | Present and forced to zero |

A comparable target-only run is:

```bash
python3 PROS/fastgpro_based_pros/train_pros.py \
  --config PROS/fastgpro_based_pros/configs/pros_multitask_fastgrpo_example.json \
  --generation-backend target \
  --train-draft false \
  --output-dir outputs/pros_multitask_target \
  --log-file outputs/pros_multitask_target/train.jsonl
```

Keep the model, data/task config, seed, batch/repeat counts, sampling parameters,
maximum length, reward setup, objective, and optimizer settings identical when
comparing the two backends.

## JSONL Events and Metrics

`log_file` is strict JSON Lines: every line is serialized with non-finite values
replaced by finite zeros and can be parsed independently with `json.loads`.
Every event receives a monotonic `tb_step`, UTC `timestamp`, and
`elapsed_time_sec`, plus `epoch`, actor `step`/`global_step`,
`generation_attempt`, and `generation_backend` context. The startup event uses
epoch and step zero.

| Event | Emission rule | Main contents |
| --- | --- | --- |
| `config` | Once after initialization | Resolved config plus effective LoRA/speculative settings. |
| `generation` | After every generation attempt, including no-valid-prompt and all-actor-skipped attempts | Counts, rewards, lengths, phase timing, generation performance, diagnostics, tasks, and PROS tree/sampler metrics. |
| `train` | Only when actor-eligible records produce an actor update | Actor/draft losses, advantages, cumulative performance, rewards, tasks, tree/sampler state, and effective settings. |
| `eval` | At `eval_freq` actor steps when evaluation data is configured | Overall/per-task reward, accuracy, sample count, and reward diagnostics. |

Generation events include:

- prompt/completion/generated/used group counts, correct/incorrect skip counts,
  empty completions, task completion counts, and per-group decisions;
- suffix, inherited-prefix, and full-response token length count/mean/min/max,
  sample standard deviation, range, and coefficient of variation;
- generation-only wall time plus separate reward, draft-training,
  response-statistics, and tree phase timing;
- generated tokens/second and target/draft/check/prefill time values and ratios;
- emitted, accepted, verified, path-budget, and verification-round speculative
  counters, including per-round and acceptance-rate derivatives.

All ratios use safe division and remain zero for a zero denominator. Target-only
runs expose the same generation-performance schema with speculative counters and
rates set to zero.

Reward diagnostics are aggregated for the current batch, cumulatively, and per
task. They include all completions, even groups excluded from actor training, and
track pass/fail rates, timeouts, missing tests/entry points, parse status, format
reward, reward/error/test-type counts, group decisions, and ignored-group error
counts. `reward_debug_sample_count` caps failed examples and
`reward_debug_sample_chars` bounds prompt, completion, stdout, and stderr text.

PROS-specific metrics include:

- `dataset/num_nodes` and partial-rollout length/ratio statistics;
- `sampler/selected_thetas_*`, posterior correlation/error, diversity threshold,
  fallback totals, and per-task quota/selection/shortfall/fallback maps;
- `actor/loss`, policy-gradient/KL/clip-fraction and advantage metrics;
- `draft/loss1`, `draft/loss2`, and draft/actor phase times;
- cumulative and per-task reward, length, completion, skip, and usage metrics.

### TensorBoard and Progress

With `use_tensorboard=true`, finite scalar leaves from every event are mirrored
to TensorBoard under the event name. Task/tag names are sanitized, and large
lists/text are not recursively logged. If `tensorboard_log_dir` is empty, logs
go to a `tensorboard/` directory beside the JSONL file. Missing or failed
TensorBoard support emits a warning and training continues with JSONL only.

When `tqdm` is installed, the finite attempt budget drives the progress bar. Its
postfix reports actor step, compact prompt-level task composition, reward,
generated tokens/second, and skipped groups. Detailed task composition remains
available in `task_prompt_counts`, `task_completion_counts`, and `task_metrics`.
The bar advances for actor-skipped attempts and is closed during trainer cleanup.

## PROS Prefix Window for Short Code

`tree_min_window_tokens` defaults to `1000`, matching the port's existing PROS
configuration. A successful rollout creates a child only when its valid prefix
selection window is longer than this threshold. Short code completions therefore
often train normally but create no augmented child nodes at the default value.

For a code-heavy experiment, explicitly lower the threshold after inspecting
your response-length distribution, for example:

```bash
python3 PROS/fastgpro_based_pros/train_pros.py \
  --config PROS/fastgpro_based_pros/configs/pros_multitask_fastgrpo_example.json \
  --tree-min-window-tokens 128
```

This is an experiment-level algorithmic choice; the package default and bundled
trainer examples remain `1000`.

## Original-PROS-Like Objective Settings

The default objective is `fastgrpo` for a fair comparison with FastGRPO. For a
configuration closer to the original PROS GRPO run:

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

## Explicit Non-Goals

This PROS package does **not** implement C_peak measurement, verification-capacity
benchmarking, capacity sweeps, or capacity autotuning. `verification_capacity`
is an explicit user-provided speculative-decoding control. FastGRPO-specific
benchmark scripts are intentionally not imported or mirrored here.

## Migration Map

| Reference component | Local component | Decision |
| --- | --- | --- |
| Original PROS tree engine and PG posterior | `pros_tree.py::ProsTreeEngine` | Reimplemented without ray/verl; retains prefix selection, hierarchical posterior, and diversity. |
| Original PROS partial-rollout prompt/mask flow | `pros_trainer.py::EncodedPrompt` and `TrainingRecord.new_token_mask` | Reimplemented for FastGRPO tokenization and target loss. |
| Original PROS GRPO/GPG algorithms | `pros_loss.py` | Reimplemented with `fastgrpo`, `clipped_grpo`, and `gpg` objectives. |
| FastGRPO multi-task/reward/code helpers | `FastGRPO/helper/multitask.py`, `rewards.py`, and `code_verifier.py` | Reused directly; the executable `grpo_speculative.py` is not imported. |
| FastGRPO speculative generation/model helpers | `FastGRPO/helper/specualtive_generate.py` and `modeling_draft.py` | Reused through PROS orchestration. |
| PROS observability | `pros_logging.py` | Strict JSONL, optional TensorBoard, finite metrics, and bounded reward diagnostics. |

## Verification Commands

```bash
python3 -m unittest discover -s FastGRPO/tests -v
python3 -m unittest discover -s PROS/fastgpro_based_pros/tests -v
python3 PROS/fastgpro_based_pros/train_pros.py \
  --dry-run true \
  --verification-capacity 256 \
  --max-draft-token-length 6 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj \
  --lora-bias none \
  --train-draft false
```

GPU/model end-to-end validation is separate from these model-free checks and
should only be claimed after running with already available local weights.
