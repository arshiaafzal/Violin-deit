#!/bin/bash
#SBATCH --job-name=test2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --account=a-a06

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model violin_tiny_pos_cls_trial2 --batch-size 256 --data-path $SCRATCH/imagenet --output_dir $SCRATCH/output2