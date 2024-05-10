# apt update && apt upgrade -y 没啥必要
apt update
apt install -y tmux zsh nvtop fish htop tldr fish
pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple 
pip uninstall -y transformer-engine # transformers
pip install -U "black[jupyter]" isort ipython rich huggingface-hub bpytop transformers datasets accelerate -i https://mirrors.aliyun.com/pypi/simple  sentencepiece
pip install peft==0.3.0
pip install flash-attn --no-build-isolation 
# torch 版本是flash_attn要求的
python -c 'import transformers;import accelerate;import flash_attn'
