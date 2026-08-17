#!/bin/bash
#SBATCH -A research
#SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH -p kcis
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodelist gnode054
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=v_397.txt
# 6, 12, 19, 26, 31, 33, 37, 42, 45, 47, 48, 49, 50, 52, 57, 59, 60, 62, 63, 67, 69, 71, 74, 79, 82, 84, 85, 86, 92

# export XLA_PYTHON_CLIENT_PREALLOCATE=False
# export CUDA_VISIBLE_DEVICES=0
# export JAX_PLATFORM_NAME=cpu
# python launcher/viz/viz_map.py 
# python sweep.py
# JAX_TRACEBACK_FILTERING=off
# python train_offline.py --config train_config.py:r --project 130826_PR --mode 1 --env_id 29 --transfer True
python viz_map.py --model_location 'results/PointRobot/397/'
# 0,1,4,5,6,7,16, 17, 19, 21 to 28, metadrive 30 to 38 ,pr 29



