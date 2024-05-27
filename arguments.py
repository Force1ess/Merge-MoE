from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import TrainingArguments
import os


# default setting follow dora-llama7b
@dataclass
class TraningArguments(TrainingArguments):
    model_name_or_path: Optional[str] = field(default="./smol_llama-4x220M-MoE")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "The implementation of attention mechanism."},
    )
    data_workers: int = field(
        default=min(os.cpu_count() // torch.cuda.device_count() // 2, 32),
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). "
            "0 means that the data will be loaded in the main process."
        },
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether to pin memory for faster data transfer."},
    )
    output_dir: Optional[str] = field(
        default="saved_models",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    save_safetensors: bool = field(
        default=False,
        metadata={"help": "Whether to save safetensors."},
    )
    weight_decay: float = field(
        default=0.1,
        metadata={"help": "Weight decay for optimizer."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup ratio."},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps."},
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."},
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Total number of training epochs to perform."},
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "The initial learning rate for Adam."},
    )


@dataclass
class DataArguments:
    path: Optional[str] = field(
        default="./minipile",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    split: Optional[str] = field(
        default="validation[:1024]",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library)."
        },
    )
    keep_in_memory: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to keep the dataset in memory. This allows for faster training."
        },
    )


# text brewer setting lambda = 0.9 and router aux loss weight - 0.01
@dataclass
class DistillArguments:
    distill_config: str = field(
        default="configs/all_kd_config.json",
        metadata={"help": "The path to the distillation config file."},
    )
    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=64, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    expert_merge: str = field(
        default="keep1", metadata={"help": "Merge method for MoE."}
    )
    expert_init: str = field(
        default=None, metadata={"help": "Initialization method for MoE."}
    )
    intermediate_loss: str = field(
        default="hidden_mse",
        metadata={"help": "The loss function for intermediate loss."},
    )
