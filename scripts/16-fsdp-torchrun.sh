torchrun --nproc_per_node=2 main.py \
    --num_train_epochs 1 \
    --save_only_model True\
    --per_device_train_batch_size 1 \
    --distill_config ./configs/mini_distill_config.json \
    --logging_steps 5 \
    --model_max_length 2048 \
    --bf16 True\
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'EVEMixtralDecoderLayer' 
    # --tf32 True \
    # --gradient_checkpointing True \
    # --gradient_accumulation_steps 16 \

