#!/bin/bash
# Launch script for distributed GRPO training

# Set number of GPUs
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
echo "Launching training on $NUM_GPUS GPUs"

# Optional: Set specific GPUs to use
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    -m hud.rl.train \
    "$@"

# Example usage:
# ./launch_ddp_training.sh --config config.json --test
# NUM_GPUS=2 ./launch_ddp_training.sh --config config.json
