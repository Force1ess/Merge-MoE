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
    --output_dir 'saved_models/test'\
    --per_device_train_batch_size 4\
    --distill_config $1\
    --bf16 True \
    --tf32 True \
    --attn_implementation "flash_attention_2" 
    #--deepspeed $2
    #--bf16 True \
