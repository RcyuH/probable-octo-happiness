WORK_DIR="/workspace/storage-shared/nlp/huypq51/projects/AAAI27_no_name/probable-octo-happiness-main/FastGRPO"


python3 ${WORK_DIR}/train_draft.py \
    --model_dir /workspace/storage-shared/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --version_name draft_debug \
    --model_type deepseek \
    --batch_size 16 \
    --num_epochs 10 \
    --lr 5e-5 \
    --accumulation_steps 16 \
    --warmup_ratio 0.05 \
    --sample_num 100 \
    --log_dir ${WORK_DIR}/logs/draft/DeepSeek-R1-Distill-Qwen-1.5B \
    --saved_model_dir ${WORK_DIR}/ckpts/draft/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset_dir ${WORK_DIR}/data/gsm8k_pretrain_draft/train.json