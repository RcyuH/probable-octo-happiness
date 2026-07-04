
# FastGRPO

[FastGRPO](https://arxiv.org/abs/2509.21792) is an **adaptive speculative decoding framework** for Group Relative Policy Optimization (GRPO) that dynamically adjusts drafting and verification strategies based on real-time concurrency levels. The framework addresses the prohibitively slow training process of GRPO by maximizing acceleration of the generation phase while maintaining reasoning capabilities.

## 📋 Overview

The key innovations of this project include:

1. **Adaptive Speculative Decoding**: Dynamically adjusts drafting and verification strategy based on real-time concurrency levels, maximizing the acceleration of the generation process
2. **Joint Draft Model Training**: Mitigates performance degradation caused by distributional drift between the evolving target model and draft model through continuous adaptation using feedback from the target model
3. **Significant Speedup**: Achieves an end-to-end speedup of 2.35× to 2.72× compared to baseline approaches

### Core Components

1. **`train_draft.py`**: Pre-training script for the draft model
   - Used for initial draft model training
   - Can be used standalone for draft model preparation

2. **`grpo_speculative.py`**: Main training script for the GRPO speculative decoding framework
   - Jointly trains target and draft models
   - Implements speculative decoding during training
   - Accelerates GRPO training process without performance loss



## 🛠️ Prerequisites
 
Before starting training, please complete the following setup steps:

### Step 1: Environment Setup
Install the required dependencies via the provided `requirement.txt`:
```bash
pip install -r requirement.txt
```

### Step 2: Dataset Preparation
Download and place the training dataset under the `data/` directory.  
Ensure the data is properly formatted and accessible. Example:
```
data/
├── simplelr_abel_level3to5
├── gsm8k
└── ...
```


## 🚀 Usage

### GRPO Speculative Training (Joint Training)

Launch the joint training process for both target and draft models:

```bash
python train_draft.py \
    --model_dir <path_to_pretrained_model> \
    --version_name <your_experiment_name> \
    --model_type qwen2 \
    --batch_size 1 \
    --num_epochs 10 \
    --lr 5e-5 \
    --accumulation_steps 16 \
    --warmup_ratio 0.05 \
    --sample_num 100 \
    --log_dir <path_to_training_log_dir> \
    --saved_model_dir <dir_to_save_model_checkpoints> \
    --dataset_dir <dir_to_dataset>
```

```bash
python grpo_speculative.py \
    --model_dir <path_to_target_model> \                                  
    --adapter_path <path_to_pretrained_draft_adapter> \                  
    --load_lora_path <path_to_resume_checkpoint_or_empty> \               
    --model_type qwen2 \                                      
    --train_option simplelr_abel_level3to5 \                          
    --version_name debug \                            
    --batch_size 4 \
    --num_epochs 10 \
    --sample_num 100 \
    --accumulation_steps 4 \
    --draft_accumulation_steps 1 \
    --target_lr 1e-6 \
    --draft_lr 1e-4 \
    --lora_r 64 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_bias none \
    --is_train_draft True \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_length 2048 \
    --verification_capacity 160 \
    --max_draft_token_length 5 \
    --min_draft_token_length 3 \
    --max_draft_k 8 \
    --max_verification_num 160 \
    --draft_token_length_c 0.75 \
    --max_training_padding_gap 256 \
    --max_training_token 3072 \
    --grpo_iteration_num 1 \
    --repeated_generate_nums 8 \
    --beta 0.04 \
    --epsilon 0.1 \
    --log_file <path_to_save_training_log> \
    --use_tensorboard True \
    --tensorboard_log_dir <path_to_tensorboard_dir> \
    --saved_model_dir <dir_to_save_target_model_checkpoints> \           
    --saved_draft_model_dir <dir_to_save_draft_model_checkpoints> \       
    --saved_statistics_dir <dir_to_save_generation_length_stats> \       
```

### Multi-task RLVR Benchmarking

The default training path is still the original math-only setup. To benchmark
FastGRPO on a multi-task RLVR mixture, pass a task config with `--task_config`.
The script then keeps the same GRPO/speculative decoding logic, but routes each
sample through task-aware prompting, reward dispatch, and per-task logging.

Example config: `configs/multitask_rlvr.example.json`

Supported built-in reward types:
- `math_latex`: math verifier reward plus optional format reward
- `exact_match`: normalized exact string match
- `contains`: normalized substring match
- `regex`: regex match through `pattern`
- `format_only`: checks the `<think>` closing tag format used by the original script
- `code`: placeholder coding reward; extracts code, checks Python syntax when `language`
  is Python, checks optional `entry_point`, and checks optional
  `expected_substrings`
- `code_unit_test`: executes generated Python in a local subprocess against the
  configured `tests`; supports assert-style tests, `unittest.TestCase` classes,
  pytest-style `test_*` functions, and `test_type: "stdin_stdout"` cases with
  `{"input": "...", "output": "..."}` records. Configure `timeout_seconds`
  per task or record.
- `zero`: always returns 0

You can also set `custom_reward_func` to a Python callable path such as
`my_rewards.code_reward`. The callable receives `completion=` and `example=`.
Use `code_unit_test` for built-in coding-task evaluation, or use this hook when
you need a hardened external sandbox. The built-in `code` reward does not
execute generated code. `code_unit_test` runs generated code on the local
training machine with a timeout; use it only in a trusted environment.
For math datasets, `answer_extraction: "gsm8k_hash"` extracts the text after
`####`, and `answer_template: "\\boxed{{{answer}}}"` wraps the verifier target.
For coding datasets, set `prompt_type: "code"` and optionally provide
`language`, `entry_point_field`, `tests_field`, `test_type`, `timeout_seconds`,
`starter_code_field`, and `expected_substrings_field`.

FastGRPO run:

```bash
python grpo_speculative.py \
    --model_dir <path_to_target_model> \
    --adapter_path <path_to_pretrained_draft_adapter> \
    --task_config configs/multitask_rlvr.example.json \
    --generation_backend speculative \
    --model_type qwen2 \
    --version_name multitask_fastgrpo \
    --batch_size 4 \
    --num_epochs 1 \
    --repeated_generate_nums 8 \
    --target_lr 1e-6 \
    --lora_r 64 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_bias none \
    --verification_capacity 160 \
    --max_draft_token_length 5 \
    --min_draft_token_length 3 \
    --max_draft_k 8 \
    --max_verification_num 160 \
    --draft_token_length_c 0.75 \
    --is_train_draft True \
    --log_file <path_to_fastgrpo_log> \
    --use_tensorboard True \
    --tensorboard_log_dir <path_to_tensorboard_dir> \
    --saved_model_dir <dir_to_save_target_checkpoints> \
    --saved_draft_model_dir <dir_to_save_draft_checkpoints> \
    --saved_statistics_dir <dir_to_save_generation_stats>
```

Target-only baseline with the same harness:

```bash
python grpo_speculative.py \
    --model_dir <path_to_target_model> \
    --adapter_path <path_to_pretrained_draft_adapter> \
    --task_config configs/multitask_rlvr.example.json \
    --generation_backend target \
    --model_type qwen2 \
    --version_name multitask_target_baseline \
    --batch_size 4 \
    --num_epochs 1 \
    --repeated_generate_nums 8 \
    --is_train_draft False \
    --log_file <path_to_baseline_log> \
    --saved_model_dir <dir_to_save_target_checkpoints> \
    --saved_draft_model_dir <dir_to_save_draft_checkpoints> \
    --saved_statistics_dir <dir_to_save_generation_stats>
```

The target policy is trained with PEFT LoRA by default. `--target_lr` controls
the LoRA adapter optimizer, while `--lora_r`, `--lora_alpha`,
`--lora_dropout`, `--lora_target_modules`, and `--lora_bias` are passed directly
to `peft.LoraConfig`. The defaults match the previous hard-coded setup:

```bash
--lora_r 64 \
--lora_alpha 32 \
--lora_dropout 0.0 \
--lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
--lora_bias none
```

The JSON logs include `generation_backend`, `lora_config`, and `task_metrics`,
so compare `generate_time_cost`, `train_time_cost`, `mean_reward`,
`average_acc_length`, and the per-task skip counts between these two runs.

TensorBoard scalar logs are enabled by default when the `tensorboard` package is
installed. If `--tensorboard_log_dir` is omitted, event files are written to a
`tensorboard/` directory beside `--log_file`.

```bash
tensorboard --logdir <path_to_tensorboard_dir>
```

The log stream now emits both `generation` events and `train` events.
Generation events are written even when every group is skipped and `used_items`
does not increase. Reward diagnostics are available under `reward_debug`,
`reward_debug_batch`, and each `task_metrics.<task_id>.reward_debug`, including:
- `error_type_counts`: verifier/reward outcomes such as `none`, `empty_tests`,
  `empty_completion`, `syntax`, `runtime`, `assertion`, `wrong_answer`,
  `timeout`, `import`, and `unsupported_language`
- `ignored_incorrect_error_type_counts`: error types among completions in groups
  skipped as `ignore_due_incorrect`
- `ignored_correct_error_type_counts`: outcomes among groups skipped as
  `ignore_due_correct`
- `missing_tests_count`, `missing_entry_point_count`, `pass_rate`,
  `mean_extracted_code_chars`, and `test_type_counts`

Speculative decoding metrics are logged under `generation_perf`:
- `speculative_avg_emitted_tokens_per_round`: average number of output tokens
  emitted per target verification round; this is the explicit version of the
  older `average_acc_length` metric
- `speculative_avg_accepted_draft_tokens_per_round`: average accepted draft
  tokens per verification round, excluding the guaranteed target token
- `speculative_path_acceptance_rate`: accepted draft tokens divided by the
  draft-path token budget; this is usually the most intuitive acceptance-rate
  metric
- `speculative_tree_acceptance_rate`: accepted draft tokens divided by all draft
  tokens verified in the draft tree
- `speculative_verified_draft_tokens_per_round`, `speculative_verification_rounds`,
  `speculative_accepted_draft_tokens`, and `generated_tokens_per_second`


### Speculative Generate Function Parameters

The speculative_generate function is the core function of our project. The following is an introduction to its parameters:

#### Basic Parameters
| Parameter | Type | Description |
|----------|------|-------------|
| `input_ids` | tensor | Input token IDs for the generation process (shape: [batch_size, seq_len]) |
| `attention_mask` | tensor | Attention mask to indicate which tokens are valid (shape: [batch_size, seq_len]) |
| `tokenizer` | tokenizer | The tokenizer associated with the model for encoding/decoding tokens |

#### Sampling Parameters
| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `do_sample` | bool | `False` | Whether to use sampling (`True`) or greedy decoding (`False`) |
| `temperature` | float | `0.8` | Sampling temperature for controlling randomness |
| `top_p` | float | `0.9` | Top-p (nucleus) sampling threshold |
| `top_k` | int | `None` | Top-k sampling parameter |

#### Adaptive Control Parameters
| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `verification_capacity` | int | `160` | Maximum capacity for verification tokens |
| `max_draft_token_length` | int | `5` | Maximum length of draft tokens to generate |
| `max_draft_k` | int | `8` | Maximum branching factor for draft tree |
| `max_verification_num` | int | `160` | Maximum number of tokens to verify |
| `min_draft_token_length` | int | `3` | Minimum length of draft tokens |
| `draft_token_length_c` | float | 0.75 | A parameter that affects the tuning of the draft token length and should be set based on the capability of the draft model; the stronger the draft model, the smaller this value should be |

In `grpo_speculative.py`, these adaptive speculative decoding parameters are
also exposed as CLI flags:

```bash
--verification_capacity 160 \
--max_draft_token_length 5 \
--min_draft_token_length 3 \
--max_draft_k 8 \
--max_verification_num 160 \
--draft_token_length_c 0.75
```

`verification_capacity` is a practical per-run budget, not a universal constant.
It should be tuned for the target model size, GPU memory/latency profile, dtype,
batch size, `repeated_generate_nums`, and max sequence length. Internally, the
active batch shares this capacity, so each sequence receives roughly
`floor(verification_capacity / active_batch_size)` verification tokens before
`max_verification_num` is applied. Increase it only if the GPU has enough
headroom and generation throughput improves.

You can measure the target model's paper-style `C_peak` with the benchmark
helper. It runs single-token target forward passes across batch sizes, computes
`throughput = batch_size / median_latency`, and returns the smallest batch size
within 95% of max throughput:

```bash
python3 FastGRPO/benchmark_verification_capacity.py \
  --model_dir /path/to/base_model \
  --benchmark_mode c_peak \
  --c_peak_b_max 256 \
  --c_peak_step 8 \
  --c_peak_warmup 10 \
  --c_peak_repeat 30 \
  --output_dir FastGRPO/benchmark_results
```

To also run the older end-to-end speculative generation sweep, pass a draft
checkpoint and use `both` mode:

```bash
python3 FastGRPO/benchmark_verification_capacity.py \
  --model_dir /path/to/base_model \
  --adapter_path /path/to/draft_model.pt \
  --benchmark_mode both \
  --capacities 64,96,128:320:32 \
  --batch_size 4 \
  --repeated_generate_nums 8 \
  --num_batches 3 \
  --warmup_batches 1 \
  --max_length 2048 \
  --output_dir FastGRPO/benchmark_results
```

The benchmark writes `*_c_peak.csv/json` for the forward-pass measurement. In
`both`/`capacity` mode it also writes per-batch JSONL plus summary CSV/JSON files
and recommends the fastest successful capacity by generated tokens per second.
Use `--capacities c_peak` to run the generation sweep only at the measured
`C_peak`, or use `--task_config`, `--train_option`, `--prompt_file`, or repeated
`--prompt` flags to benchmark prompts that match your real workload.
#### Output Control Parameters
| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `repeated_generate_nums` | int | `None` | Number of repeated generations for each input |
| `statistical_time` | bool | `True` | Whether to collect timing statistics |
| `return_all_draft_input` | bool | `False` | Whether to return all draft inputs |
| `max_length` | int | `2048` | Maximum length of generated sequences |


## Reference

```
@misc{zhang2025fastgrpoacceleratingpolicyoptimization,
      title={FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning}, 
      author={Yizhou Zhang and Ning Lv and Teng Wang and Jisheng Dang},
      year={2025},
      eprint={2509.21792},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.21792}, 
}
```
