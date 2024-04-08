# -*- encoding: utf-8 -*-
# here put the import lib
from ast import Not
import importlib
from typing import Callable
from dataclasses import dataclass, field

from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
    MixtralSparseMoeBlock,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D

from ..utils import (
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    TRANSFORMERS_MODELS_TO_EVELORA_TARGET_MODULES_MAPPING,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
)


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


@dataclass
class EVELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MMOELora`]
    """

    # expert_num: int = field(default=8)

    def __init__(
        self,
        merge_method: Callable[
            [nn.ModuleList, nn.ModuleList], MixtralBlockSparseTop2MLP
        ],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.merge_method = merge_method

    def __post_init__(self):
        self.peft_type = PeftType.EVELORA


class EVELoraModel(LoraModel):
    """
    Create MMOELoRA (MMOE based LoRA) model from a pretrained transformers model.
    """

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
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit:
            raise NotImplementedError(
                "EVELoraModel does not support 8-bit model. Please set `is_loaded_in_8bit` to False."
            )
        is_target_modules_in_base_model = False
        lora_args = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            target_module_found = any(
                key.endswith(target_key) for target_key in lora_config.target_modules
            )
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                if isinstance(target, EVELoraExpert):
                    raise ValueError(
                        "Target module cannot be a LoRA layer. Please check the target modules and try again."
                    )
                else:
                    if loaded_in_8bit:
                        raise NotImplementedError
                    else:
                        if isinstance(target, MixtralSparseMoeBlock):
                            ffn_dim, hidden_dim = (
                                target.ffn_dim,
                                target.hidden_dim,
                            )
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `MixtralSparseMoeBlock` is supported."
                            )
                        # lora layer 和 sparse layer只在这里被调用
                        new_module = EVEMixtralSparseBlock(
                            adapter_name,
                            target.top_k,
                            lora_args,
                            ffn_dim,
                            hidden_dim,
                            target.gate,
                            target.experts,
                            lora_config.merge_method,
                        )

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
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


# modulelist can only contain nn.module
class EVELoraExpert(LoraLayer, nn.Module):

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        device,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights:bool = True
    ):
        nn.Module.__init__(self)
        LoraLayer.__init__(self, in_features, out_features)

        self.weight = torch.tensor(0, device=device)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        # placehold for lora layer update

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype

        if self.r[self.active_adapter] > 0 and not self.merged:
            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            return (self.lora_B[self.active_adapter](
                self.lora_A[self.active_adapter](
                    self.lora_dropout[self.active_adapter](x)
                )
            ) * self.scaling[self.active_adapter]).to(
                previous_dtype
            )  # TODO scaling?
        else:
            raise NotImplementedError("EVELoraExpert does not support such operation.")

    def merge(self):
        raise ValueError("EVELoraExpert does not support merge operation.")

    def unmerge(self):
        raise ValueError("EVELoraExpert does not support unmerge operation.")

# ? 目前的lora只模拟了w2的lora
class EVEMixtralSparseBlock(nn.Module):
    def __init__(
        self,
        adapter_name: str,
        num_experts_per_tok: int,
        lora_config: dict,
        ffn_dim: int,
        hidden_dim: int,
        router: nn.Linear,
        experts: nn.ModuleList,
        merge_method: Callable[
            [nn.ModuleList, nn.ModuleList], MixtralBlockSparseTop2MLP
        ],
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = len(experts)
        self.ffn_dim, self.hidden_dim = ffn_dim, hidden_dim
        self.adapter_name = adapter_name
        self.router = router
        device = experts[0].w1.weight.device
        self.lora_experts = nn.ModuleList(
            [
                EVELoraExpert(adapter_name, self.hidden_dim,self.hidden_dim, device,**lora_config)
                for _ in range(self.num_experts)
            ]
        )

        self.expert = merge_method(experts, self.lora_experts)
        self.expert.requires_grad_(False)
        self.merged = False
        # TODO soft router for future test
        self.expert_weight = [0] * self.num_experts

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        raise NotImplementedError("EVEMixtralSparseBlock does not support update_layer operation.")
        # assert r > 0, "r should be greater than 0"
        # self.lora_experts = {adapter_name: self.lora_experts}
        # for i in range(self.num_experts):
        #     self.lora_experts[i].update_layer(
        #         adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
        #     )

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_experts.keys():
            for i in range(self.num_experts):
                self.lora_experts[i].reset_lora_parameters(adapter_name)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = (
                self.expert(current_state) * routing_weights[top_x_list, idx_list, None]
            )
            lora_expert = self.lora_experts[expert_idx]
            current_hidden_states+=lora_expert.forward(current_state)
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits
