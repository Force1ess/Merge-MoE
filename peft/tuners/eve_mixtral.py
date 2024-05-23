import torch
import torch.nn.functional as F
from torch import nn
from peft.tuners.eve_layers import EVETop2MLP
from typing import Optional, Tuple
import warnings


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
        # TODO 由于使用了share_expert 这里可以使用1？
        self.top_k = num_experts_per_tok
        self.num_experts = len(experts)
        self.ffn_dim, self.hidden_dim = ffn_dim, hidden_dim
        self.adapter_name = adapter_name
        self.router = router
        device = experts[0].w1.weight.device
        dtype = experts[0].w1.weight.dtype
        self.eve_experts = nn.ModuleList(
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
        for i in range(self.eve_experts):
            self.eve_experts[i].reset_lora_parameters()

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
            eve_expert = self.eve_experts[expert_idx]
            # TODO 是不是这里写错了，因为share_expert没有一个自己的weight，这里似乎应该rescale一下
            current_hidden_states = (
                eve_expert.forward(current_state)
                * routing_weights[top_x_list, idx_list, None]
            )
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        # copy from qwen without share expert gate and sigmoid
        shared_expert_output = self.share_expert(hidden_states)
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class EVEMixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size,
        self_attn,
        sparse_moe_block,
        eve_sparse_moe_block,
        input_layernorm,
        post_attention_layernorm,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = self_attn
        self.sparse_moe_block = sparse_moe_block
        self.eve_sparse_moe_block = eve_sparse_moe_block
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

        sparse_hidden_states, _ = self.sparse_moe_block(hidden_states)
        self.sparse_hidden_states = residual + sparse_hidden_states

        eve_hidden_states, router_logits = self.eve_sparse_moe_block(hidden_states)
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
