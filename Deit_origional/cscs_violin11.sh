#!/bin/bash
#SBATCH --job-name=v11
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --account=a-a06
#SBATCH --output=slurm_outs/v11_%j.log

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model violin_tiny --batch-size 256 --data-path $SCRATCH/imagenet --output_dir $SCRATCH/output1 \
                                                                        --pos_emb --mask fixed --scale --method mul_v1