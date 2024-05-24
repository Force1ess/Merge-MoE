from torch import nn
from peft.utils.other import dequantize_module_weight, transpose
import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN
import math

class EVETop2MLP(nn.Module):
    def __init__(
        self,
        ffn_dim: int,
        hidden_dim: int,
        act_fn: str,
        dtype,
        device,
        eve_args: dict,
    ):
        nn.Module.__init__(self)
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim
        self.w1 = EVELinear(
            self.hidden_dim, self.ffn_dim, dtype=dtype, device=device, **eve_args
        )
        self.w2 = EVELinear(
            self.ffn_dim, self.hidden_dim, dtype=dtype, device=device, **eve_args
        )
        self.w3 = EVELinear(
            self.hidden_dim, self.ffn_dim, dtype=dtype, device=device, **eve_args
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
            result = (self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling).to(
                previous_dtype
            )  # TODO scaling?
            return result

        else:
            raise NotImplementedError("EVELinear does not support such operation.")

    def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, channel-wise
        weight = weight + scaling * lora_weight
        # the following is needed to have compatibility with the 4D weight tensors of Conv2D
        weight_norm = weight.norm(p=2, dim=(1, 2, 3), keepdim=True).transpose(1, 0)
        return weight_norm


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
