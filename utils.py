import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import torch
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
