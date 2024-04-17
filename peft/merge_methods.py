from torch import nn


def keep1(experts: nn.ModuleList, lora_experts: nn.ModuleList):
    return experts[0]


MERGE_MAP = {
    "keep1": keep1,
}
