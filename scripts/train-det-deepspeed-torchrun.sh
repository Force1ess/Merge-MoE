CUDA_DEVICES_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "CUDA_DEVICES_COUNT: $CUDA_DEVICES_COUNT"
export PYTHONPATH=.
while true; do
    random_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    if ! lsof -i:$random_port &> /dev/null; then
        break
    fi
done
torchrun --master_port $random_port --nproc_per_node $CUDA_DEVICES_COUNT main.py \
    --num_train_epochs 1 \
    --save_only_model True \
    --bf16 True \
    --per_device_train_batch_size 8\
    --model_max_length 4096\
    --distill_config $1\
    --logging_steps 5\
    --save_strategy "epoch" \
    --attn_implementation "flash_attention_2" \
    --gradient_accumulation_steps 8
    #--deepspeed $2
