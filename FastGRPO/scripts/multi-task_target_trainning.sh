WORK_DIR="/workspace/storage-shared/nlp/huypq51/projects/AAAI27_no_name/probable-octo-happiness-main/FastGRPO"

CUDA_VISIBLE_DEVICES=0 python3 ${WORK_DIR}/grpo_speculative.py \
    --model_dir /workspace/storage-shared/models/DeepSeek-R1-Distill-Qwen-1.5B  \
    --adapter_path ${WORK_DIR}/ckpts/draft/DeepSeek-R1-Distill-Qwen-1.5B/step1168.pth \
    --task_config ${WORK_DIR}/configs/multitask_rlvr.json \
    --generation_backend speculative \
    --model_type deepseek \
    --version_name multitask_fastgrpo_debug \
    --batch_size 4 \
    --num_epochs 1 \
    --repeated_generate_nums 8 \
    --is_train_draft True \
    --log_file ${WORK_DIR}/logs/running/multi-task-target/DeepSeek-R1-Distill-Qwen-1.5B \
    --saved_model_dir ${WORK_DIR}/ckpts/multi-task-target/DeepSeek-R1-Distill-Qwen-1.5B \
    --saved_draft_model_dir ${WORK_DIR}/ckpts/multi-task-target/DeepSeek-R1-Distill-Qwen-1.5B \
    --saved_statistics_dir ${WORK_DIR}/logs/stats/multi-task-target/DeepSeek-R1-Distill-Qwen-1.5B 