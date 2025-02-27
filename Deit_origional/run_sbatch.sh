#!/bin/bash  
#SBATCH --job-name=test  
#SBATCH --time=24:00:00  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --gpus-per-node=4  
#SBATCH --cpus-per-task=8  
#SBATCH --mem=256G  
#SBATCH --account=a-a06

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29581 --use_env main.py --model violin_base_patch16_224 --batch-size 256  --data-path "/iopsstor/scratch/cscs/aafzal/imagenet" --output_dir "./output/Violin_decay_aug"

