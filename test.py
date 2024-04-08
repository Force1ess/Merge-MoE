import torch
torch.distributed.init_process_group(backend='nccl')
print(torch.cuda.current_device())
