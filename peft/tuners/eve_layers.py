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
        use_dora: bool = False,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
    ):
        nn.Module.__init__(self)
        self.use_dora = use_dora
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
            if self.use_dora:
                result += self._apply_dora(x, self.lora_A, self.lora_B, self.scaling)
            return result

        else:
            raise NotImplementedError("EVELinear does not support such operation.")

    def dora_init(self) -> None:
        lora_A = self.lora_A.weight
        lora_B = self.lora_B.weight
        scaling = self.scaling
        # TODO base_layer?
        base_layer = self.get_base_layer()
        weight = dequantize_module_weight(base_layer)
        if weight.data.ndim == 4:  # For handling LoRAs applied to Conv2Ds.
            raise NotImplementedError(
                "DORA is not supported for Conv2D layers. Please set `use_dora` to False."
            )
        else:
            lora_weight = lora_B @ lora_A
        # TODO DORA 施工中
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)

        self.lora_magnitude_vector = nn.Parameter(weight_norm, requires_grad=True)

    def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, channel-wise
        weight = weight + scaling * lora_weight
        # the following is needed to have compatibility with the 4D weight tensors of Conv2D
        weight_norm = weight.norm(p=2, dim=(1, 2, 3), keepdim=True).transpose(1, 0)
        return weight_norm

    def _apply_dora(self, x, lora_A, lora_B, scaling):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        lora_weight = lora_B.weight @ lora_A.weight
        magnitude = self.lora_magnitude_vector
        base_layer = self.get_base_layer()
        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        result_dora = (mag_norm_scale - 1) * (
            F.linear(x, transpose(weight, self.fan_in_fan_out))
        ) + mag_norm_scale * lora_B(lora_A(x)) * scaling

        # Note: Computation could potentially be accelerated by using the code below instead of calculating X@W again.
        # This is only correct if dropout=0, otherwise results will differ:
        # https://github.com/huggingface/peft/pull/1474#issuecomment-1964682771
        # bias = self.get_base_layer().bias
        # if bias is not None:
        #     result = result - bias
        # result = mag_norm_scale * result + mag_norm_scale * lora_B(lora_A(x)) * scaling
        # if bias is not None:
        #     result = result + bias

        return result_dora

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
        if self.use_dora:
            self.dora_init()

    def reset_lora_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def merge(self):
        raise ValueError("EVELoraExpert does not support merge operation.")

    def unmerge(self):
        raise ValueError("EVELoraExpert does not support unmerge operation.")
