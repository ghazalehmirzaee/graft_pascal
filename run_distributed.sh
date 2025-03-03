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

# Set environment variables to prevent timeout issues
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800  # 30 minutes timeout

# Set PyTorch specific variables
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Set OMP variables for better performance
export OMP_NUM_THREADS=1

# project directory
cd /NewRaidData/ghazal/pascal/graft_pascal

# Run training script with torch.distributed.launch
# Using 4 GPUs instead of 8 to reduce risk of NCCL timeouts
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --config config/default.yaml \
    --start_phase phase2_finetune \
    --experiment_name "finetune_distributed"

