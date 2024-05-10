# sync cur folder with /shared_home/zhenghao2022/Merge-MoE
default:
	echo "Please specify a target: make push"
sync:
	rsync -avzh /home/zhenghao2022/Merge-MoE/{思路记录.md,install.sh,main.py,modeling_mixtral.py,scripts,utils.py,arguments.py,kd_trainer.py,makefile,peft,test.ipynb,configs,textbrewer} /shared_home/zhenghao2022/Merge-MoE/
pull:
	rsync -avzh /shared_home/zhenghao2022/Merge-MoE/{思路记录.md,install.sh,main.py,modeling_mixtral.py,scripts,utils.py,arguments.py,kd_trainer.py,makefile,peft,test.ipynb,configs,textbrewer}  /home/zhenghao2022/Merge-MoE
archive:
	zip mergemoe.zip -r *py scripts configs peft textbrewer .vscode
stop:
	docker stop /moguozhao_zhuquebenchmark; docker rm /moguozhao_zhuquebenchmark
connect:
	docker attach moguozhao_zhuquebenchmark