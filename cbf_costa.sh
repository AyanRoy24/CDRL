#!/bin/bash
#SBATCH -A research
#SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH -p kcis
#SBATCH -n 10
#SBATCH --gres=gpu:1
# SBATCH --nodelist gnode092    
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
# SBATCH --output=results/outputFiles/T_p4.txt
#SBATCH --output=chevel_cbf_B25.txt
# 6, 12, 19, 26, 31, 33, 37, 42, 45, 47, 48, 49, 50, 52, 57, 59, 60, 62, 63, 67, 69, 71, 74, 79, 82, 84, 85, 86, 92

# export XLA_PYTHON_CLIENT_PREALLOCATE=False
# export CUDA_VISIBLE_DEVICES=0
# export JAX_PLATFORM_NAME=cpu
# python launcher/viz/viz_map.py 
# python sweep.py
# JAX_TRACEBACK_FILTERING=off

# "ant-dir-safe",                          # 41
# "ant-goal-safe",                         # 42
# "ant-circle-safe",                       # 43
# "ant-walk-safe",                         # 44
# "half-cheetah-vel-safe",                 # 45
# "half-cheetah-walk-safe",                # 46


python train_general_costa.py --config train_config_costa.py:r --src_id 45 --src_goal "0,1,2" --eval_budgets "25" --project 041126

# python adapt_flow.py --config train_config_costa.py:r --project 311026 --src_id 42 --tgt_goal 2 --train_seed 194   # omit --train_seed for the newest run

# python adapt_finetune.py --config train_config_costa.py:r --project 311026 --src_id 42 --tgt_goal 2 --eval_budgets "10,20,40,80" --parts cost --train_seed 194

# python adapt_online.py --config train_config_costa.py:r --project 311026 --src_id 42 --tgt_goal 0 --train_seed 194 --eval_budgets "10,20,40,80"  

# python tune_pooled.py --n-trials 60  --confirm 3
