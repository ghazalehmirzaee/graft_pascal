#!/bin/bash
#SBATCH --job-name=graft-pascal
#SBATCH --output=graft-pascal_%j.out
#SBATCH --error=graft-pascal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --mem=187G

# Load required modules
module load cuda/12.2
module load anaconda3

# Activate your conda environment
source activate myenv

# project directory
cd /NewRaidData/ghazal/pascal/graft_pascal

# Run training script with torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    train.py \
    --config config/default.yaml \
    --start_phase phase2_finetune \
    --experiment_name "finetune_distributed"
