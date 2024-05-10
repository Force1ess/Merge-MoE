# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import math
from typing import Callable, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from ..merge_methods import MERGE_MAP
import warnings

from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
    MixtralSparseMoeBlock,
    MixtralDecoderLayer,
    MixtralModel,
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
class EVELoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MMOELora`]
    """

    # expert_num: int = field(default=8)
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_method: str = field(
        default=None, metadata={"help": "Merge method for MixtralSparseMoeBlock"}
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
        MixtralModel.forward = mixtral_monkey_patch_forward
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
        merge_method = MERGE_MAP[eve_config.merge_method]
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            target_module_found = str.isdigit(key.replace("model.layers.", ""))
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
                            merge_method,
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


# modulelist can only contain nn.module
class EVELoraExpert(nn.Module):
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
        # placehold for lora layer update

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
            self.lora_A = nn.Linear(self.in_features, r, bias=False, dtype=self.dtype)
            self.lora_B = nn.Linear(r, self.out_features, bias=False, dtype=self.dtype)
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
        eve_config: dict,
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
        dtype = experts[0].w1.weight.dtype
        self.lora_experts = nn.ModuleList(
            [
                EVELoraExpert(
                    self.hidden_dim,
                    self.hidden_dim,
                    dtype,
                    device,
                    **eve_config,
                )
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
                self.expert(current_state) + lora_expert.forward(current_state)
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

        block_sparse_hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        block_sparse_hidden_states = residual + block_sparse_hidden_states

        eve_hidden_states, _ = self.eve_block_sparse_moe(hidden_states)
        self.eve_hidden_states = residual + eve_hidden_states

        hidden_states = block_sparse_hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


from transformers.models.mixtral.modeling_mixtral import (
    DynamicCache,
    logger,
    Cache,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
    MoeModelOutputWithPast,
)


def mixtral_monkey_patch_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, MoeModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    past_key_values_length = 0

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if (
        attention_mask is not None
        and self._attn_implementation == "flash_attention_2"
        and use_cache
    ):
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_router_logits = () if output_router_logits else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                output_router_logits,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_router_logits:
            all_router_logits += (layer_outputs[-1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_router_logits,
            ]
            if v is not None
        )
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_logits,
    )
