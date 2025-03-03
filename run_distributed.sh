#!/bin/bash
#SBATCH --job-name=graft-pascal
#SBATCH --output=graft-pascal_%j.out
#SBATCH --error=graft-pascal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --mem=187G

# Use direct path to CUDA instead of module load
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Activate your conda environment
source activate myenv

# Project directory
cd /NewRaidData/ghazal/pascal/graft_pascal

# Run training script with torchrun
torchrun \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    train.py \
    --config config/default.yaml \
    --start_phase phase2_finetune \
    --experiment_name "finetune_distributed"

