from utils import (
    prepare_teacher_cache,
    get_cache_folder,
)
from arguments import (
    TraningArguments,
    DataArguments,
    DistillArguments,
)
from peft import get_peft_model, EVELoraConfig, TaskType
from textbrewer import TrainingConfig, DistillationConfig, GeneralDistiller
import torch
import os
import merge_methods
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


def simple_adaptor(batch, model_outputs):
    # The second and third elements of model outputs are the logits and hidden states
    return {"logits": model_outputs["logits"], "hidden": model_outputs["hidden_states"],
            "losses":model_outputs['loss']
            }


# TODO logits mask? len(hidden)=23
# TODO losses 是什么？
# TODO 测试多卡推理
def main():
    # parse arguments and prepare dataset, model, tokenizer, dataloader
    training_args, data_args, distill_args = transformers.HfArgumentParser(
        (TraningArguments, DataArguments, DistillArguments)
    ).parse_args_into_dataclasses()
    training_args.local_rank = int(os.getenv("LOCAL_RANK", -1))
    #device = None
    if training_args.local_rank != -1:
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(training_args.local_rank)
        print(f"local rank: {training_args.local_rank}/ init ")
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=16,
    )
    cache_exist, cache_dir = get_cache_folder(
        training_args.cache_dir,
        training_args.model_name_or_path,
        data_args.path,
        training_args.batch_size,
    )
    teacher_model = None
    if cache_exist != True:
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
        sampler = None
        if training_args.local_rank != -1:
            sampler = DistributedSampler(dataset)

        if training_args.do_cache:
            prepareloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=training_args.batch_size,
                num_workers=training_args.data_workers,
                pin_memory=training_args.pin_memory,
                sampler=sampler,
                collate_fn = lambda x:x
            )
            prepare_teacher_cache(prepareloader, data_collator,training_args, cache_dir)
        else:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                training_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            teacher_model.eval()
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=training_args.batch_size,
                num_workers=training_args.data_workers,
                pin_memory=training_args.pin_memory,
                collate_fn=data_collator,
                sampler=sampler,
            )
    else:
        dataloader = load_dataset("parquet", data_dir=cache_dir)

    eve_config = EVELoraConfig(
        task_type=TaskType.CAUSAL_LM,
        merge_method=getattr(merge_methods, distill_args.merge_method),
        r=distill_args.lora_r,
        lora_alpha=distill_args.lora_alpha,
        lora_dropout=distill_args.lora_dropout,
    )
    student_model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ),
        eve_config,
    )
    student_model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=training_args.learning_rate
    )

    train_config = TrainingConfig(
        device=student_model.device,
        local_rank=training_args.local_rank,
    )

    # Distillation configuration
    # Matching different layers of the student and the teacher
    # 所有22个层都是一样的
    # 会不会是我的causal lm 没有label的问题
    distill_config = DistillationConfig(
        hard_label_weight=distill_args.hard_label_weight,
        temperature=distill_args.distill_temperature,
        intermediate_matches=[
            {
                "layer_T": 0,
                "layer_S": 0,
                "feature": "hidden",
                "loss": "hidden_mse",
                "weight": 1,
            },
            {
                "layer_T": 8,
                "layer_S": 8,
                "feature": "hidden",
                "loss": "hidden_mse",
                "weight": 1,
            },
        ],
    )

    # Build distiller
    # batch processor可以传一个
    # 初始化ddp等
    distiller = GeneralDistiller(
        train_config=train_config,
        distill_config=distill_config,
        model_T=teacher_model,
        model_S=student_model,
        adaptor_T=simple_adaptor,
        adaptor_S=simple_adaptor,
    )

    # 内有逻辑，若使用torchrun启动会使用DDP
    with distiller:
        distiller.train(
            optimizer,
            dataloader,
            num_epochs=training_args.num_train_epochs,
            callback=None,
            output_hidden_states=True,
        )
    # TODO save

if __name__ == "__main__":
    main()
