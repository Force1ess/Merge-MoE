# sync cur folder with /shared_home/zhenghao2022/Merge-MoE
default:
	echo "Please specify a target: make push"
rm:
	rm -rf teacher_cache
sync:
	rsync -avzh /home/zhenghao2022/Merge-MoE/ /shared_home/zhenghao2022/Merge-MoE/