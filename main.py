import json
import os
import socket
from datetime import datetime
from pathlib import Path

import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)

from arguments import DataArguments, DistillArguments, TraningArguments
from kd_trainer import KDTrainer
from peft import EVEConfig, TaskType, get_peft_model
from textbrewer import DistillationConfig
from utils import dir_check, rank0_print, send_feishu


def main():
    training_args, data_args, distill_args = transformers.HfArgumentParser(
        (TraningArguments, DataArguments, DistillArguments)
    ).parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
    )
    tokenizer.padding_side='left'
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=16,
    )
    dataset = load_dataset(**vars(data_args))
    dataset = dataset.map(
        lambda x: {
            "input_ids": tokenizer(
                x["text"],
                truncation=True,
                max_length=training_args.model_max_length,
                padding=False,
            )["input_ids"]
        },
        keep_in_memory=data_args.keep_in_memory,
        num_proc=training_args.data_workers,
        remove_columns=dataset.column_names,
    )

    eve_config = EVEConfig(
        task_type=TaskType.CAUSAL_LM,
        expert_merge=distill_args.expert_merge,
        expert_init=distill_args.expert_init,
        r=distill_args.lora_r,
        lora_alpha=distill_args.lora_alpha,
        lora_dropout=distill_args.lora_dropout,
    )
    config = AutoConfig.from_pretrained(training_args.model_name_or_path)
    if data_args.split.startswith("train"):
        model = AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            attn_implementation=training_args.attn_implementation,
        )
    else:
        config.num_hidden_layers = 1
        model = AutoModelForCausalLM.from_config(config)

    if data_args.split == "train":
        training_args.report_to = ["wandb"]
    else:
        training_args.report_to = []

    eve_model = get_peft_model(
        model,
        eve_config,
    )
    eve_model.print_trainable_parameters()
    while (
        training_args.per_device_train_batch_size
        * torch.cuda.device_count()
        * training_args.gradient_accumulation_steps
        < 64
    ):
        training_args.gradient_accumulation_steps *= 2

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        datetime.now().strftime("%m-%d-%H-%M-%S")
        + f"{training_args.model_name_or_path}-{Path(distill_args.distill_config).stem}".replace(
            "/", "-"
        )
        + distill_args.expert_merge
        + "-"
        + distill_args.expert_init,
    )
    rank0_print(f'model saved to -> {training_args.output_dir}')
    dir_check(training_args.output_dir)
    distill_config: dict = json.load(open(distill_args.distill_config, "r"))
    if distill_config['intermediate_loss_weight']=='layer':
        distill_config['intermediate_loss_weight']=1/config.num_hidden_layers
    sep_intermediate_layers = distill_config.pop("sep_intermediate_layers", 1)
    distill_config = DistillationConfig.from_dict(
        distill_config
        | {
            "intermediate_matches": [
                {
                    "layer_T": i,
                    "layer_S": i,
                    "loss": distill_args.intermediate_loss,
                    "feature": "hidden_states",
                    "weight": 1,
                }
                for i in range(config.num_hidden_layers - 1)
                if i % sep_intermediate_layers == 0
            ]
        }
    )

    trainer = KDTrainer(
        eve_model,
        training_args,
        distill_config,
        data_collator,
        dataset,
        tokenizer=tokenizer,
    )
    if data_args.split == "train":
        rank0_print(
            f"total batch_size: {training_args.per_device_train_batch_size*torch.cuda.device_count()*training_args.gradient_accumulation_steps}"
        )
        rank0_print(
            f"weight: [0]label {distill_config.hard_label_weight} [1]logits {distill_config.kd_loss_weight} [2]inter {distill_config.intermediate_loss_weight}"
        )
    trainer.train()
    if os.environ.get("LOCAL_RANK", "0") == "0" and data_args.split == "train":
        send_feishu(
            f"{socket.gethostname()}: 训练完成，模型保存在{training_args.output_dir}"
        )


if __name__ == "__main__":
    main()
