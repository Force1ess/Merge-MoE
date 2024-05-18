from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import TrainingArguments
import os


@dataclass
class TraningArguments(TrainingArguments):
    model_name_or_path: Optional[str] = field(default="./Mini-Mixtral-v0.2")
    model_max_length: int = field(
        default=2048,
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


@dataclass
class DataArguments:
    path: Optional[str] = field(
        default="./minipile",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    split: Optional[str] = field(
        default="validation[:128]",
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


@dataclass
class DistillArguments:
    distill_config: str = field(
        default="configs/all_kd_config.json",
        metadata={"help": "The path to the distillation config file."},
    )
    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
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
