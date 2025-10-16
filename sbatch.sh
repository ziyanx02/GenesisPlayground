#!/bin/bash
#SBATCH -J GSPlayground          # Job name
#SBATCH -p gpu-preempt           # Partition
#SBATCH -G 1                     # Request 1 GPU
#SBATCH -c 8                     # Request 8 CPUs
#SBATCH --mem=20G                # Memory
#SBATCH -t 480                   # Time (minutes)
#SBATCH -C l40s                  # GPU constraint
#SBATCH -o logs/%x-%j.out        # Save stdout to logs/jobname-jobid.out
#SBATCH --nodes=1

# Load environment
source /scratch4/workspace/junyunhuang_umass_edu-myworkspace/GenesisPlayground/.venv/bin/activate

# Move to project folder
cd /scratch4/workspace/junyunhuang_umass_edu-myworkspace/GenesisPlayground

export WANDB_ENTITY="h975894552"
export WANDB_PROJECT="genesis"

exp_name=${1:-None}
python /scratch4/workspace/junyunhuang_umass_edu-myworkspace/GenesisPlayground/examples/run_ppo_walking.py \
    --exp_name $exp_name \
    # --reward_args.OrientationPenalty 10 \
    # --reward_args.LinVelZPenalty 2 \
    # --reward_args.G1FeetHeightPenalty 100 \
    # --reward_args.G1FeetContactForcePenalty 30 \
python /scratch4/workspace/junyunhuang_umass_edu-myworkspace/GenesisPlayground/examples/run_ppo_walking.py \
    --eval True \
    --show_viewer False \
    --exp_name $exp_name
