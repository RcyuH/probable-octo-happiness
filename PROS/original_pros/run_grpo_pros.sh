# for qwen3
VAL_TEMP=0.6
VAL_TOPP=0.95
VAL_TOPK=20

# Model settings
USE_OVERLONG=True
SAMPLER=tree # null, tree
DATA_WORKERS=0
CRITIC_WARMUP=0
ROLLOUT_N=8
BATCH_SIZE=512
MINI_BSZ=64
OVERLONG_BUFFER_LEN=$((1024 * 1))
OVERLONG_COEF=1
MAX_PROMPT_LEN=$((1024 * 1))
MAX_RESPONSE_LEN=$((1024 * 5 + OVERLONG_BUFFER_LEN))
CLIP_HIGHER=0.28

# Tree Selector
TREE_SELECTOR=entropy # entropy mix
NUM_GIBBS=100
GIBBS_DISCOUNT=0.99

# Performance tuning
N_NODES=32
N_GPUS=16
ROLLOUT_TP_SIZE=1
OFFLOAD=True
TOTAL_EPOCHS=400
FORWARD_RATIO=10
BACKWARD_RATIO=3
FORWARD_MAX_TOKEN_LEN=$((FORWARD_RATIO * (MAX_PROMPT_LEN + MAX_RESPONSE_LEN)))
BACKWARD_MAX_TOKEN_LEN=$((BACKWARD_RATIO * (MAX_PROMPT_LEN + MAX_RESPONSE_LEN)))


BASE_MODEL=${MY_MODEL_DIR}/Qwen/Qwen3-8B-Base
CRITIC_MODEL=${MY_MODEL_DIR}/Qwen/Qwen3-8B-Base


TEMPLATE_TYPE=chat
TRAIN_FILE="${MY_DATA_DIR}/DAPO-Math-17k/train.parquet"
TEST_FILES="${MY_DATA_DIR}/merged_math_datasets/merged_test.parquet"

PROJ_NAME="PROS"
MODEL_NAME=$(basename $BASE_MODEL)
DATA_NAME=DAPOTrain
EXPERIMENT_NAME="${DATA_NAME}_GPRO"

python3 examples/data_preprocess/custom.py \
    --resume


# set -x
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="."

CMD="python3 -m verl.trainer.main_ppo \
    data.sampler.name=$SAMPLER \
    data.dataloader_num_workers=${DATA_WORKERS} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGHER} \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=\"$TEST_FILES\" \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.sampler.tree_sampler.gibbs_sweeps=${NUM_GIBBS} \
    data.sampler.tree_sampler.gamma=${GIBBS_DISCOUNT} \
    data.tree_data.name=${TREE_SELECTOR} \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BSZ \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$BACKWARD_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$FORWARD_MAX_TOKEN_LEN \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$FORWARD_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOPK} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOPP} \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward_model.launch_reward_fn_async=True \
    reward_model.overlong_buffer.enable=${USE_OVERLONG} \
    reward_model.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
    reward_model.overlong_buffer.penalty_factor=${OVERLONG_COEF} \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir=$MY_CKPT_DIR/$PROJ_NAME/$EXPERIMENT_NAME"


eval $CMD

