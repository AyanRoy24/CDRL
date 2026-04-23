#!/bin/bash
#SBATCH -A research
#SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH -p kcis
#SBATCH -n 10
#SBATCH --gres=gpu:1
# SBATCH --nodelist gnode056
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=SafetyAntVelocityEnv.txt
# 6,7, 12, 26, 31, 33, 37, 42, 45, 47, 48, 49, 50, 52, 56, 57, 59, 60, 62, 63, 67, 69, 71, 74, 82, 84, 85, 86, 92
# export XLA_PYTHON_CLIENT_PREALLOCATE=False
# export CUDA_VISIBLE_DEVICES=0
# export JAX_PLATFORM_NAME=cpu
# python sweep.py
# python train_md.py --mode 7 --project 210126_MD --config train_config.py:r  --max_steps 500001 --cost_tau 0.35 --reward_tau 0.75 --env_id 32
# 38, 35, 32 - dense  
# 37, 34, 31 - mean
# 36, 33, 30 - sparse
# xvfb-run -s "-screen 0 1400x900x24" python launcher/examples/train_md.py --mode=1 --project='201025' --config configs/train_config.py:r --env_id 35

python examples/customized/collect_dataset.py