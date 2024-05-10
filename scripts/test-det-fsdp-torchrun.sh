#!/bin/bash

CUDA_DEVICES_COUNT=$(nvidia-smi --list-gpus | wc -l)

# 尝试获取一个未被使用的随机端口
while true; do
    random_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    # 检查端口是否被占用（使用 lsof 或 netstat）
    if ! lsof -i:$random_port &> /dev/null; then
        break
    fi
done

torchrun --master_port $random_port --nproc_per_node $CUDA_DEVICES_COUNT main.py \
    --save_only_model True \
    --per_device_train_batch_size 4 \
    --distill_config $1\
    --logging_steps 20 \
    --model_max_length 2048 \
    --bf16 True \
    --split "validation" \
    --save_strategy "steps" \
    --save_steps "2500" \
    --save_total_limit 5 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --fsdp_transformer_layer_cls_to_wrap 'EVEMixtralDecoderLayer'
    # --tf32 True \
    # --gradient_checkpointing True \
    # --gradient_accumulation_steps 16 \
