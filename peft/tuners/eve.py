# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import math
from typing import Callable, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from ..merge_methods import init_experts
import warnings

from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
    MixtralDecoderLayer,
    ACT2FN,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """
    This is the configuration class to store the configuration of a [`~peft.MMOELora`]
    """

    # expert_num: int = field(default=8)
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
        default=None, metadata={"help": "Target modules for EVELora"}
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
        self.peft_type = PeftType.EVELORA


class EVEModel(LoraModel):
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
        # MixtralModel.forward = mixtral_monkey_patch_forward
        eve_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit:
            raise NotImplementedError(
                "EVELoraModel does not support 8-bit model. Please set `is_loaded_in_8bit` to False."
            )
        is_target_modules_in_base_model = False
        lora_args = {
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

                else:
                    raise ValueError(
                        f"Target module {target} is not supported. "
                        f"Currently, only `MixtralDecoderLayer` is supported."
                    )
                # lora layer 和 sparse layer只在这里被调用
                # TODO 两个layer
                # 实现merge等
                sparse_moe_module = EVEMixtralSparseBlock(
                    adapter_name,
                    target.block_sparse_moe.top_k,
                    lora_args,
                    ffn_dim,
                    hidden_dim,
                    target.block_sparse_moe.gate,
                    target.block_sparse_moe.experts,
                )
                sparse_moe_module.share_expert = init_experts(
                    target.block_sparse_moe.experts,
                    sparse_moe_module.lora_experts,
                    lora_args,
                    eve_config.expert_merge,
                    eve_config.expert_init,
                )
                new_module = EVEMixtralDecoderLayer(
                    target.hidden_size,
                    target.self_attn,
                    target.block_sparse_moe,
                    sparse_moe_module,
                    target.input_layernorm,
                    target.post_attention_layernorm,
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


class EVETop2MLP(nn.Module):
    def __init__(
        self,
        ffn_dim: int,
        hidden_dim: int,
        act_fn: str,
        dtype,
        device,
        lora_args: dict,
    ):
        nn.Module.__init__(self)
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim
        self.w1 = EVELinear(
            self.hidden_dim, self.ffn_dim, dtype=dtype, device=device, **lora_args
        )
        self.w2 = EVELinear(
            self.ffn_dim, self.hidden_dim, dtype=dtype, device=device, **lora_args
        )
        self.w3 = EVELinear(
            self.hidden_dim, self.ffn_dim, dtype=dtype, device=device, **lora_args
        )
        self.act_fn = ACT2FN[act_fn]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(
            hidden_states
        )
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class EVELinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype,
        device,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
    ):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device
        self.update_layer(r, lora_alpha, lora_dropout, init_lora_weights)
        self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype

        if self.r > 0 and not self.merged:
            return (self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling).to(
                previous_dtype
            )  # TODO scaling?
        else:
            raise NotImplementedError("EVELoraExpert does not support such operation.")

    def update_layer(self, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout = lora_dropout_layer
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(
                self.in_features,
                r,
                dtype=self.dtype,
                device=self.device,
            )
            self.lora_B = nn.Linear(
                r,
                self.out_features,
                dtype=self.dtype,
                device=self.device,
            )
            self.scaling = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters()
        self.to(self.device)

    def reset_lora_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

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
        lora_args: dict,
        ffn_dim: int,
        hidden_dim: int,
        router: nn.Linear,
        experts: nn.ModuleList,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = len(experts)
        self.ffn_dim, self.hidden_dim = ffn_dim, hidden_dim
        self.adapter_name = adapter_name
        self.router = router
        device = experts[0].w1.weight.device
        dtype = experts[0].w1.weight.dtype
        self.lora_experts = nn.ModuleList(
            [
                EVETop2MLP(
                    self.ffn_dim,
                    self.hidden_dim,
                    "silu",
                    dtype,
                    device,
                    lora_args,
                )
                for _ in range(self.num_experts)
            ]
        )
        
        self.merged = False
        # TODO soft router for future test
        self.expert_weight = [0] * self.num_experts

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        raise NotImplementedError(
            "EVEMixtralSparseBlock does not support update_layer operation."
        )

    def reset_lora_parameters(self, adapter_name):
        for i in range(self.lora_experts):
            self.lora_experts[i].reset_lora_parameters()

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
            lora_expert = self.lora_experts[expert_idx]
            current_hidden_states = (
                self.share_expert(current_state) + lora_expert.forward(current_state)
            ) * routing_weights[top_x_list, idx_list, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class EVEMixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size,
        self_attn,
        block_sparse_moe,
        eve_block_sparse_moe,
        input_layernorm,
        post_attention_layernorm,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = self_attn
        self.block_sparse_moe = block_sparse_moe
        self.eve_block_sparse_moe = eve_block_sparse_moe
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        block_sparse_hidden_states, _ = self.block_sparse_moe(hidden_states)
        self.block_sparse_hidden_states = residual + block_sparse_hidden_states

        eve_hidden_states, router_logits = self.eve_block_sparse_moe(hidden_states)
        eve_hidden_states = residual + eve_hidden_states

        hidden_states = eve_hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
