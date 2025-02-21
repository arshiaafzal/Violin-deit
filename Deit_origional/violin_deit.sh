CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29539 --use_env main.py  --model violin_base_pos_cls \
    --batch-size 512  \
    --data-path "/home/data_shared/imagenet/Data/CLS-LOC" --output_dir "./output/violin_deit" \

