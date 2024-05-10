import torch
import pandas as pd
import os
import sys
import requests
import time
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
print = tqdm.write
from transformers import AutoModelForCausalLM

model = None
device = None
hostname = os.uname()[1]
rank = os.getenv("RANK", None)

def rank0_print(text):
    if rank is None or int(rank) == 0:
        print(text)

def dir_check(dir_path: str, overwrite: bool = True):
    if overwrite:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)  # 防止多个进程同时创建
        return dir_path

    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            logging.warninging(f"{dir_path} have been created, but it is not a dir")
        elif len(os.listdir(dir_path)) != 0:
            orig_path = dir_path
            dir_path = orig_path + time.strftime("%m-%d-%H-%M")
            logging.warninging(
                f"{orig_path} have been created, and it is not empty, thus dir_path moved to {dir_path}"
            )
        else:
            return dir_path
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

def send_feishu(msg):
    msg = str(msg)
    headers = {"Content-Type": "application/json"}
    data = {
        "msg_type": "text",
        "content": {"text": hostname + ": " + msg + "".join(sys.argv)},
    }
    response = requests.post(
        "https://www.feishu.cn/flow/api/trigger-webhook/01db450d719418d75f29a1b637cf2ca4",
        json=data,
        headers=headers,
    )
    return response.json()


# GB
def count_avail_gpu_mem() -> int:
    return (
        torch.cuda.get_device_properties(device).total_memory
        - torch.cuda.memory_allocated(device)
    ) // 1024**3


def tensor_transform(tensor: torch.Tensor):
    return tensor.detach().view(-1).to("cpu", non_blocking=True).float().numpy()


# 各个rank的数据是分开收的
def save_cache(results, filename):
    torch.cuda.synchronize()  # 等待所有tensor 移动完成
    pd.DataFrame.from_dict(results).to_parquet(filename)


# 不能在这里转换为cpu
# 这里是异步的，所以也不应该在这里处理
def teacher_infer(batch) -> dict:
    batch = {k: v.to(device) for k, v in batch.items()}
    output = model(**batch, output_hidden_states=True)
    return output


def get_cache_folder(cache_dir, model_name, datapath, batch_size):
    cache_dir = f"{cache_dir}/{model_name.replace('/', '-')}-bs{batch_size}-{datapath.split('/')[-1].replace('/', '-')}"
    if not os.path.exists(cache_dir) or os.listdir(cache_dir) == []:
        os.makedirs(cache_dir, exist_ok=True)
        return False, cache_dir
    return True, cache_dir


# 先只存5层的试一下
def prepare_teacher_cache(dataloader, collate_fn, training_args, cache_dir):
    global model
    global device
    print(f"prepare teacher_cache save to {cache_dir}, args: {training_args}")
    # 不能使用auto
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name_or_path, torch_dtype=torch.bfloat16
    )
    device = model.device
    if training_args.local_rank != -1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        device = device = torch.cuda.current_device()
        model.to(device)  #! torch ddp 之前必须先move过来
        model = DDP(model, device_ids=[device], output_device=device)
    model.eval()
    futures = []
    executor = ThreadPoolExecutor(max_workers=training_args.data_workers)
    results = []
    rank_id = (
        "rank_" + str(training_args.local_rank)
        if training_args.local_rank != -1
        else ""
    )
    with torch.inference_mode():
        for idx, input_ids in enumerate(tqdm(dataloader)):
            batch = collate_fn(input_ids)
            output = teacher_infer(batch)
            result = {
                "input_ids": input_ids,  # batch_size, seq_len
                "logits": tensor_transform(output.logits),
                # batch_size, seq_len, vocab_size
                **{
                    f"hidden_states_{i}": tensor_transform(h)
                    for i, h in enumerate(output.hidden_states)
                    if i % 7 == 0  # ? 只保存7的倍数的hidden_states
                },
            }
            results.append(result)

            if (
                len(results) % int(training_args.cache_step // training_args.batch_size)
                == 0
                or idx == len(dataloader) - 1
            ):
                # 提交异步保存任务
                filename = f"{cache_dir}/{rank_id}step{len(results)}.parquet"
                futures.append(executor.submit(save_cache, results, filename))
                results = []

            # 效果非常好，显存占用几乎没有增加
            if count_avail_gpu_mem() < 8:
                # 内存不足，等待一半的异步任务完成
                middle = len(futures) // 2
                wait_futures, futures = futures[:middle], futures[middle:]
                for future in as_completed(wait_futures):
                    future.result()  # 这里可以处理异常或获取结果

    for future in as_completed(futures):
        future.result()  # 这里可以处理异常或获取结果

    executor.shutdown(wait=True)  # 确保所有任务完成后关闭执行器
