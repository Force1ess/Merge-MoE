# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
from dataclasses import dataclass, field
from peft.tuners.merge_methods import init_experts

from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from peft.tuners.eve_layers import EVETop2MLP

from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer

from peft.tuners.eve_mixtral import EVEMixtralDecoderLayer, EVEMixtralSparseBlock
from peft.tuners.eve_qwen import EVEQwen2MoeDecoderLayer, EVEQwen2MoeSparseMoeBlock

import torch.nn as nn

from peft.utils.config import PeftConfig

from ..utils import (
    PeftType,
    _get_submodules,
    TRANSFORMERS_MODELS_TO_EVELORA_TARGET_MODULES_MAPPING,
)

from .lora import (
    LoraModel,
    mark_only_lora_as_trainable,
)


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


@dataclass
class EVEConfig(PeftConfig):
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    expert_merge: str = field(
        default="mean", metadata={"help": "Merge method for share expert use mergekit"}
    )
    expert_init: str = field(
        default=None, metadata={"help": "eve expert initialization method"}
    )
    target_modules: list = field(
        default=None, metadata={"help": "Target modules for EVE"}
    )
    init_lora_weights: bool = field(
        default=True, metadata={"help": "Whether to initialize Lora weights"}
    )
    bias: str = field(
        default="none", metadata={"help": "Whether to use bias in the adapter"}
    )
    inference_mode: bool = field(
        default=False, metadata={"help": "Whether to use inference mode"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.EVE


class EVEModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:  # get the lora config
            model_config = (
                self.model.config.to_dict()
                if hasattr(self.model.config, "to_dict")
                else self.model.config
            )
            config = self._prepare_evelora_config(config, model_config)  # load config
            self.peft_config[adapter_name] = config  # subsititue the original config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 or self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "EVELoraModel supports only 1 adapter and it should not have bias, set bias to 'none' for the adapter."
            )

        # mark lora 之后就没有grad了
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            raise ValueError(
                "EVELoraModel does not support inference mode. Please set `inference_mode` to False."
            )

    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        # MixtralModel.forward = mixtral_monkey_patch_forward
        eve_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit:
            raise NotImplementedError(
                "EVELoraModel does not support 8-bit model. Please set `is_loaded_in_8bit` to False."
            )
        is_target_modules_in_base_model = False
        eve_args = {
            "r": eve_config.r,
            "lora_alpha": eve_config.lora_alpha,
            "lora_dropout": eve_config.lora_dropout,
            "init_lora_weights": eve_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            target_module_found = str.isdigit(key.replace("model.layers.", ""))
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)

                if isinstance(target, MixtralDecoderLayer):
                    ffn_dim, hidden_dim = (
                        target.block_sparse_moe.ffn_dim,
                        target.block_sparse_moe.hidden_dim,
                    )
                    eve_sparse_moe_module = EVEMixtralSparseBlock(
                        adapter_name,
                        target.block_sparse_moe.top_k,
                        eve_args,
                        ffn_dim,
                        hidden_dim,
                        target.block_sparse_moe.gate,
                        target.block_sparse_moe.experts,
                    )
                    eve_sparse_moe_module.share_expert = init_experts(
                        target.block_sparse_moe.experts,
                        eve_sparse_moe_module.eve_experts,
                        eve_args,
                        ["w1", "w2", "w3"],
                        eve_config.expert_init,
                        eve_config.expert_merge,
                    )
                    new_module = EVEMixtralDecoderLayer(
                        target.hidden_size,
                        target.self_attn,
                        target.block_sparse_moe,
                        eve_sparse_moe_module,
                        target.input_layernorm,
                        target.post_attention_layernorm,
                    )

                elif isinstance(target, Qwen2MoeDecoderLayer):
                    config = target.self_attn.config
                    layer_idx = target.self_attn.layer_idx
                    if (
                        layer_idx + 1
                    ) % config.decoder_sparse_step != 0 or config.num_experts < 1:
                        continue
                    eve_sparse_moe_module = EVEQwen2MoeSparseMoeBlock(config)
                    experts = target.mlp.experts
                    device = experts[0].gate_proj.weight.device
                    dtype = experts[0].gate_proj.weight.dtype
                    eve_experts = nn.ModuleList(
                        [
                            EVETop2MLP(
                                config.moe_intermediate_size,
                                config.hidden_size,
                                "silu",
                                dtype,
                                device,
                                eve_args,
                            )
                            for _ in range(self.num_experts)
                        ]
                    )
                    init_experts(
                        experts,
                        eve_experts,
                        eve_args,
                        ["gate_proj", "down_proj", "up_proj"],
                        eve_config.expert_init,
                        None,
                    )
                    eve_sparse_moe_module.experts = eve_experts
                    new_module = EVEQwen2MoeDecoderLayer(
                        eve_sparse_moe_module,
                        config,
                        layer_idx
                    )

                else:
                    raise ValueError(
                        f"Target module {target} is not supported. "
                        f"Currently, only `MixtralDecoderLayer` is supported."
                    )

                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {eve_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_evelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if (
                model_config["model_type"]
                not in TRANSFORMERS_MODELS_TO_EVELORA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = (
                TRANSFORMERS_MODELS_TO_EVELORA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        if peft_config.inference_mode:
            raise NotImplementedError(
                "EVELoraModel does not support inference mode. Please set `inference_mode` to False."
            )
        return peft_config
