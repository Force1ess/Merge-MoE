import torch
from typing import Optional, Tuple
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeConfig,
    Qwen2MoeSparseMoeBlock,
    Qwen2MoeDecoderLayer,
)


# just alter the experts property
class EVEQwen2MoeSparseMoeBlock(Qwen2MoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        raise NotImplementedError(
            "EVEQwen2MoeSparseMoeBlock does not support update_layer operation."
        )

    def reset_lora_parameters(self, adapter_name):
        for i in range(self.lora_experts):
            self.experts[i].reset_lora_parameters()


class EVEQwen2MoeDecoderLayer(Qwen2MoeDecoderLayer):
    def __init__(self, eve_sparse_moe_block, config: Qwen2MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.eve_sparse_moe_block = eve_sparse_moe_block

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
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

        self.sparse_hidden_states = self.mlp(hidden_states)
        hidden_states = self.eve_sparse_moe_block(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
