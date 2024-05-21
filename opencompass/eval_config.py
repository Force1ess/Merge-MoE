from mmengine.config import read_base

with read_base():
    from .configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets
    from .configs.datasets.triviaqa.triviaqa_wiki_gen_d18bf4 import triviaqa_datasets
    from .configs.datasets.gsm8k.gsm8k_gen_3309bd import gsm8k_datasets
    from .configs.datasets.humaneval.humaneval_gen_a82cae import humaneval_datasets
    from .configs.datasets.agieval.agieval_mixed_2f14ad import agieval_datasets
    from .configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314797 import (
        BoolQ_datasets,
    )
    from .configs.datasets.obqa.obqa_ppl_6aac9e import obqa_datasets
    from .configs.datasets.subjective.multiround.mtbench_pair_judge import (
        MTBenchDataset,
    )
    from .configs.summarizers.example import summarizer

    # from .configs.datasets.winogrande.winogrande_ll_c5cf57 import winogrande_datasets
    # from .configs.datasets.hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets
    # from .configs.datasets.nq.nq_open_gen_e93f8a import nq_datasets
from opencompass.models import HuggingFaceCausalLM

path = "/ciphome/zhuqiming/workspace/mindsft/sftout/llama-2-7b_sft_2024_5_17_6_2/checkpoint-5730"
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr="zhuque2-13B-dpo",
        path=path,
        tokenizer_path=path,
        tokenizer_kwargs=dict(
            padding_side="left",
            truncation_side="left",
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=4096,
        batch_size=1,
        model_kwargs=dict(device_map="npu:0", torch_dtype="torch.bfloat16"),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

datasets = sum(
    [v for k, v in locals().items() if k.endswith("_datasets") or k == "datasets"], []
)
work_dir = "./zhuque2_sft/"
