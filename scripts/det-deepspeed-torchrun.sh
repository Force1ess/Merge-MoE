CUDA_DEVICES_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "CUDA_DEVICES_COUNT: $CUDA_DEVICES_COUNT"
export PYTHONPATH=.
while true; do
    random_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    # 检查端口是否被占用（使用 lsof 或 netstat）
    if ! lsof -i:$random_port &> /dev/null; then
        break
    fi
done
torchrun --master_port $random_port --nproc_per_node $CUDA_DEVICES_COUNT main.py \
    --num_train_epochs 1 \
    --save_only_model True \
    --bf16 True \
    --per_device_train_batch_size 1\
    --distill_config $1\
    --logging_steps 50 \
    --model_max_length 2048\
    --split "train" \
    --save_strategy "epoch" \
    --report_to "wandb" \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --gradient_accumulation_steps 8\
    --attn_implementation "flash_attention_2" \
    --deepspeed $2
