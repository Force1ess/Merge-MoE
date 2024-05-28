from mmengine.config import read_base

with read_base():
    from .configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets
    from .configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_eaf81e import triviaqa_datasets
    from .configs.datasets.gsm8k.gsm8k_gen_3309bd import gsm8k_datasets
    from .configs.datasets.humaneval.humaneval_gen_a82cae import humaneval_datasets
    #from .configs.datasets.obqa.obqa_ppl_6aac9e import obqa_datasets
    from .configs.summarizers.example import summarizer

from opencompass.models import HuggingFaceCausalLM

path = "/root/Merge-MoE/smol_llama-4x220M-MoE"
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr="eve_smol_base",
        path=path,
        peft_path=None,
        tokenizer_path=path,
        tokenizer_kwargs=dict(
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=4096,
        batch_size=16,
        model_kwargs=dict(torch_dtype="torch.bfloat16"),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

datasets = sum(
    [v for k, v in locals().items() if k.endswith("_datasets") or k == "datasets"], []
)
work_dir = "./eve_models/"
