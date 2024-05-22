from functools import partial
from torch import nn
from copy import deepcopy
import torch
from mergekit.mergekit.merge_methods.generalized_task_arithmetic import (
    ConsensusMethod,
    GeneralizedTaskArithmeticMerge,
    SparsificationMethod,
)


def init_experts(
    experts: nn.ModuleList,
    lora_experts: nn.ModuleList,
    lora_args: dict,
    expert_merge: str,
    expert_init: str,
):
    merge_func = MERGE_MAP.get(expert_merge, keep_one)
    init_func = INIT_MAP.get(expert_init, lambda x, y, z: None)
    init_func(experts, lora_experts, lora_args)
    if isinstance(merge_func, GeneralizedTaskArithmeticMerge):
        merge_func = partial(TaskVectorAdaptation, task=merge_func)
    return merge_func(experts)


def keep_one(experts: nn.ModuleList):
    share_expert = deepcopy(experts[0])
    assert id(share_expert) != id(experts[0])
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


def average_merge(experts: nn.ModuleList):
    share_expert = deepcopy(experts[0])
    expert_num = len(experts)
    for key in ["w1", "w2", "w3"]:
        weights = []
        for i in range(expert_num):
            weight = getattr(experts[i], key).weight.data.clone()
            weights.append(weight)
        avg_weight = sum(weights) / expert_num
        getattr(share_expert, key).weight.data = avg_weight
    return share_expert


def TaskVectorAdaptation(
    task: GeneralizedTaskArithmeticMerge,
    experts: nn.ModuleList,
):
    share_expert = deepcopy(experts[0])
    params = {p.name: p.default_value for p in task.parameters()}
    expert_num = len(experts)
    tensor_params = {
        i: {"weight": 1 / expert_num, "density": 0.6} for i in range(expert_num)
    }
    if task.sparsification_method == SparsificationMethod.magnitude_outliers:
        tensor_params = {k: v | {"gamma": 0.01} for k, v in tensor_params.items()}
    for key in ["w1", "w2", "w3"]:
        tensors = {
            i: getattr(experts[i], key).weight.data.clone() for i in range(len(experts))
        }
        merged_weight = task.make_task(tensors, 0, params, tensor_params).execute(
            tensors
        )
        getattr(share_expert, key).weight.data = merged_weight
    return share_expert


# 配置中的权重总和不等于 1，但 normalize: true 参数会在内部自动对它们进行归一化
# 密度意味着我们只保留每个模型 50% 的参数（另一半来自基础模型）
# 通过仅保留一小部分最重要的参数（密度参数）
MERGE_MAP = {
    "keep1": keep_one,
    "average": average_merge,
    "ties": GeneralizedTaskArithmeticMerge(
        consensus_method=ConsensusMethod.sum,
        sparsification_method=SparsificationMethod.magnitude,
        default_normalize=True,
        default_rescale=False,
    ),
    "dare_ties": GeneralizedTaskArithmeticMerge(
        consensus_method=ConsensusMethod.sum,
        sparsification_method=SparsificationMethod.random,
        default_normalize=False,
        default_rescale=True,
    ),
    "dare_linear": GeneralizedTaskArithmeticMerge(
        consensus_method=None,
        sparsification_method=SparsificationMethod.random,
        default_normalize=False,
        default_rescale=True,
    ),
    "breadcrumbs": GeneralizedTaskArithmeticMerge(
        consensus_method=None,
        sparsification_method=SparsificationMethod.magnitude_outliers,
        default_normalize=False,
        default_rescale=False,
    ),
    "breadcrumbs_ties": GeneralizedTaskArithmeticMerge(
        consensus_method=ConsensusMethod.sum,
        sparsification_method=SparsificationMethod.magnitude_outliers,
        default_normalize=False,
        default_rescale=False,
    ),
}

INIT_MAP = {
    "svd": svd_decomposition,
}

if __name__ == "__main__":
    experts = nn.ModuleList(nn.Linear(10, 10) for _ in range(2))
    for i in MERGE_MAP.keys():
        merge_method: GeneralizedTaskArithmeticMerge = MERGE_MAP.get(i)
