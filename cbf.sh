#!/bin/bash
#SBATCH -A research
#SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH -p kcis
#SBATCH -n 10
#SBATCH --gres=gpu:1
# SBATCH --nodelist gnode025
#SBATCH --mem-per-cpu=6G
#SBATCH --time=4-00:00:00
#SBATCH --output=pointCircle2_s3.txt
# gpu not avl 68,66, 
# 4,6, 12, 19, 25,26, 31, 33, 37, 42, 45, 47, 48, 49, 50, 52, 57, 58, 59, 60, 62, 63, 67, 68, 69, 71, 74, 79, 82, 84, 85, 86, 92

export XLA_PYTHON_CLIENT_PREALLOCATE=False
# export CUDA_VISIBLE_DEVICES=0
# export JAX_PLATFORM_NAME=cpu
# python launcher/viz/viz_map.py 
# python sweep.py
python train_offline.py --config train_config.py:r --project 140926 --env_id 11 --cost_tau 0.15 --reward_tau 0.75 --mode 1 --max_steps 500001 --eval_budgets "10,20,40,80"
# seed 0, 10,20 
#use MSE for 0 and 1 (0.75, 0.5)
# 0,1,4,5,6,7,16, 17, 19, 21 to 28, metadrive 30 to 38 ,pr 29

# md done - 73,12,27,79,82,69,75



