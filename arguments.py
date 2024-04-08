from dataclasses import dataclass, field
from typing import Optional
import os


def check_flash_attention_2_exists():
    try:
        __import__("flash-attn")
        return "flash_attention_2"
    except ModuleNotFoundError:
        print("flash-attn not found, using default attention implementation.")
        return None


@dataclass
class TraningArguments:
    model_name_or_path: Optional[str] = field(
        default="AIChenKai/TinyLlama-1.1B-Chat-v1.0-x2-MoE"
    )
    batch_size: Optional[int] = field(default=4)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    data_workers: int = field(
        default=os.cpu_count(),
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). "
            "0 means that the data will be loaded in the main process."
        },
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether to pin memory for faster data transfer."},
    )
    do_cache: bool = field(
        default=True,
        metadata={"help": "Whether to cache the teacher outputs."},
    )
    cache_dir: str = field(
        default="teacher_cache",
        metadata={"help": "The directory to cache the teacher outputs."},
    )
    cache_step: int = field(
        default=40,
        metadata={"help": "Number of steps to cache the teacher outputs."},
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "Total number of training epochs to perform."}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for Adam."}
    )
    attn_implementation: str = field(
        default=check_flash_attention_2_exists(),
        metadata={"help": "The implementation of attention layer."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank"},
    )


@dataclass
class DataArguments:
    path: Optional[str] = field(
        default="JeanKaddour/minipile",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    split: Optional[str] = field(
        default="validation",
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
    distill_temperature: float = field(
        default=2.0, metadata={"help": "Temperature for distillation."}
    )
    hard_label_weight: float = field(
        default=0.2, metadata={"help": "Weight for hard label loss."}
    )
    lora_r: int = field(default=512, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    merge_method: str = field(
        default="keep1", metadata={"help": "Merge method for MoE."}
    )
