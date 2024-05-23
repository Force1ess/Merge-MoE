import json
import socket
import os
from kd_trainer import KDTrainer
from arguments import (
    TraningArguments,
    DataArguments,
    DistillArguments,
)
from peft import get_peft_model, EVEConfig, TaskType
from textbrewer import DistillationConfig
import torch
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoConfig,
)
from pathlib import Path
from datasets import load_dataset, load_from_disk
from utils import send_feishu, dir_check
from datetime import datetime


def main():
    training_args, data_args, distill_args = transformers.HfArgumentParser(
        (TraningArguments, DataArguments, DistillArguments)
    ).parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
    )
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
            torch_dtype=torch.bfloat16,
        )
    else:
        config.num_hidden_layers = 2
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

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        f"{training_args.model_name_or_path}-bs{training_args.per_device_train_batch_size*torch.cuda.device_count()*training_args.gradient_accumulation_steps}-{Path(distill_args.distill_config).stem}".replace(
            "/", "-"
        )
        + datetime.now().strftime("%m-%d"),
    )
    dir_check(training_args.output_dir)
    distill_config: dict = json.load(open(distill_args.distill_config, "r"))
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
        distill_config,
        training_args,
        data_collator,
        dataset,
        tokenizer=tokenizer,
    )
    if os.environ.get("LOCAL_RANK", "0") == "0" and data_args.split == "train":
        print(
            f"total batch_size: {training_args.per_device_train_batch_size*torch.cuda.device_count()*training_args.gradient_accumulation_steps}"
        )
    trainer.train()
    if os.environ.get("LOCAL_RANK", "0") == "0" and data_args.split == "train":
        send_feishu(f"{socket.gethostname()}: 训练完成，模型保存在{training_args.output_dir}")


if __name__ == "__main__":
    main()
