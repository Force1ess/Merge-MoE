# sync cur folder with /shared_home/zhenghao2022/Merge-MoE
default:
	echo "Please specify a target: make push"
rm:
	rm -rf wandb
sync:rm
	rsync -avzh /home/zhenghao2022/Merge-MoE/ /shared_home/zhenghao2022/Merge-MoE/
synccip:rm
	rsync -avzh /home/zhenghao2022/Merge-MoE/*  
pull:
	rsync -avzh /shared_home/zhenghao2022/Merge-MoE/ /home/zhenghao2022/Merge-MoE/
archive:
	zip mergemoe.zip -r .vscode *