CUDA_DEVICES_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "CUDA_DEVICES_COUNT: $CUDA_DEVICES_COUNT"
while true; do
    random_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    # 检查端口是否被占用（使用 lsof 或 netstat）
    if ! lsof -i:$random_port &> /dev/null; then
        break
    fi
done
torchrun --master_port $random_port --nproc_per_node $CUDA_DEVICES_COUNT main.py \
    --output_dir 'saved_models/test'
    --num_train_epochs 3 \
    --save_only_model True \
    --per_device_train_batch_size 8\
    --distill_config $1\
    --logging_steps 50 \
    --model_max_length 4096\
    --tf32 True \
    --split "test[:128]" \
    --save_strategy "epoch" \
    --report_to "wandb" \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --attn_implementation "flash_attention_2" \
    --deepspeed $2
    #--gradient_accumulation_steps 1 \
    #--bf16 True \
