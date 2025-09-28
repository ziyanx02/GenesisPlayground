#!/bin/bash

export WANDB_ENTITY="h975894552"
export WANDB_PROJECT="genesis"

# get first argument as experiment name, default to "walking"
exp_name=${1:-walking_FH_rew}

# training
python examples/run_ppo_legged.py \
    --exp_name "$exp_name"

# evaluation
python examples/run_ppo_legged.py \
    --eval True \
    --show_viewer False \
    --exp_name "$exp_name"