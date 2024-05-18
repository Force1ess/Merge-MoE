from heapq import merge
from torch import nn
from copy import deepcopy
import mergekit
import torch


def init_experts(
    experts: nn.ModuleList,
    lora_experts: nn.ModuleList,
    lora_args: dict,
    expert_merge: str,
    expert_init: str,
):
    merge_func = getattr(mergekit, expert_merge, lambda x,y,_:x[0])
    init_func = INIT_MAP.get(expert_init, lambda x:x)
    merge_func(experts, lora_experts, lora_args)
    init_func(experts, lora_experts, lora_args)


def keep1(experts: nn.ModuleList, lora_experts: nn.ModuleList, lora_args: dict):
    share_expert = deepcopy(experts[0])
    assert id(share_expert)!= id(experts[0])
    return share_expert


def svd_decomposition(experts, lora_experts, lora_args: dict):
    """
    Decompose a 2D matrix into low-rank matrices A and B using SVD.a

    :param weight: The matrix to decompose, of shape (H, W)
    :param lora_rank: The final rank of the decomposition
    :return: A tuple of tensors (A, B)
    """
    expert_num = len(experts)
    for i in range(expert_num):
        for key in ["w1", "w2", "w3"]:
            weight = getattr(experts[i], key).weight.data
            lora_rank = lora_args["r"]
            if weight.dim() != 2:
                raise ValueError(
                    f"Only support 2D matrix, but your input has {weight.dim()} dimensions."
                )

            # SVD Decomposition
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

            # Truncated matrices
            A = Vh[:lora_rank, :]
            B = U[:, :lora_rank] @ torch.diag(S[:lora_rank])
            lora = getattr(lora_experts[i], key)
            lora.lora_A.weight.data = A
            lora.lora_B.weight.data = B


INIT_MAP = {
    "svd": svd_decomposition,
}
