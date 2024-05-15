import json
import socket
import os
from kd_trainer import KDTrainer
from arguments import (
    TraningArguments,
    DataArguments,
    DistillArguments,
)
from peft import get_peft_model, EVELoraConfig, TaskType
from textbrewer import DistillationConfig
import torch
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
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

    eve_config = EVELoraConfig(
        task_type=TaskType.CAUSAL_LM,
        merge_method=distill_args.merge_method,
        r=distill_args.lora_r,
        lora_alpha=distill_args.lora_alpha,
        lora_dropout=distill_args.lora_dropout,
    )
    if training_args.split == "train":
        model = AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = transformers.AutoConfig.from_pretrained(
            training_args.model_name_or_path
        )
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)

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
    distill_config = DistillationConfig(
        **json.load(open(distill_args.distill_config, "r"))
    )
    trainer = KDTrainer(
        eve_model,
        distill_config,
        training_args,
        data_collator,
        dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    send_feishu(f"{socket.gethostname()}: 训练完成，模型保存在{training_args.output_dir}")


if __name__ == "__main__":
    main()
