torchrun --nproc_per_node 4 main.py \
    --batch_size 64 \
    --do_cache True\
    --cache_step 640\
    --cache_dir "/ceph_home/teacher_cache"\
    --split "train"

    
