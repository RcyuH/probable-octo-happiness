WORK_DIR="/workspace/storage-shared/nlp/huypq51/projects/AAAI27_no_name/probable-octo-happiness-main/FastGRPO"

CUDA_VISIBLE_DEVICES=0 python3 ${WORK_DIR}/grpo_speculative.py \
    --model_dir /workspace/storage-shared/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --adapter_path ${WORK_DIR}/ckpts/draft/DeepSeek-R1-Distill-Qwen-1.5B/step1168.pth \
    --load_lora_path "" \
    --model_type deepseek \
    --train_option gsm8k_train_grpo \
    --version_name target_debug \
    --batch_size 4 \
    --num_epochs 10 \
    --sample_num 100 \
    --accumulation_steps 16 \
    --draft_accumulation_steps 16 \
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
    --beta 0.04 \
    --epsilon 0.1 \
    --log_file ${WORK_DIR}/logs/running/single-task-target/DeepSeek-R1-Distill-Qwen-1.5B \
    --saved_model_dir ${WORK_DIR}/ckpts/single-task-target/DeepSeek-R1-Distill-Qwen-1.5B \
    --saved_draft_model_dir ${WORK_DIR}/ckpts/single-task-target/DeepSeek-R1-Distill-Qwen-1.5B \
    --saved_statistics_dir ${WORK_DIR}/logs/stats/single-task-target/DeepSeek-R1-Distill-Qwen-1.5B 