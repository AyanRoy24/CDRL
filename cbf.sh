#!/bin/bash
#SBATCH -A research
#SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH -p kcis
#SBATCH -n 10
#SBATCH --gres=gpu:1
# SBATCH --nodelist gnode027
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=hopVel_eval.txt

# DSRL / CoPDT pipeline: the 26 tasks of CDSRL/CoPDT/OSRL/examples/train/train_mtcdt.py
# (see dsrl_tasks.py), three stages, each reading the previous one's output:
#
#   0. per-family state/action auto-encoders     -> results/sa_encoder/<family>/
#   1. CoPDT context encoder (no discriminator)  -> results/context_encoder/default/
#   2. budget-adaptive IQL, budgets 10,20,40,80  -> results/copdt_general/<seed>/
#
# Currently run over AntVelocity v0 and v1 only (OfflineAntVelocityGymnasium-v0/-v1, read from the
# DSRL dataset folder: $DSRL_DATASET_DIR, default ../datasets, downloaded if missing).
# --tasks takes 'all' (the full 26-task suite), task names, family names or indices.

# export XLA_PYTHON_CLIENT_PREALLOCATE=False
# export WANDB_MODE=offline

# AntVelocity v0 / v2 only exist in CoPDT's fork of DSRL (the dsrl installed in the mjpy env is the
# stock one and only registers -v1, hence "Environment version v0 ... is deprecated").
export PYTHONPATH=/home2/ayanroy/CDSRL/CoPDT/DSRL:$PYTHONPATH
# The fork defaults to ../datasets (the stock dsrl to ~/.dsrl/datasets); use the already-downloaded
# DSRL folder so nothing is downloaded.
export DSRL_DATASET_DIR=$HOME/.dsrl/datasets

# TASKS="AntVel-v0,AntVel-v1"
# TASKS="HalfCheetahVel-v0,HalfCheetahVel-v1"
TASKS="HopperVel-v0,HopperVel-v1"
# TASKS="Walker2dVel-v0,Walker2dVel-v1"

# TASKS="PointButton1,PointButton2"

PROJECT="051126"
# export WANDB_MODE=offline

# python train_sa_encoder.py      --config train_config.py:r --tasks "$TASKS" --project $PROJECT
# python train_context_encoder.py --config train_config.py:r --tasks "$TASKS" --project $PROJECT --max_steps 50000
# python train_general.py         --config train_config.py:r --tasks "$TASKS" --project $PROJECT \
#     --eval_budgets "10,20,40,80"  --config.eval_episodes 10 #--config.seed 1

# export PYTHONPATH=/home2/ayanroy/CDSRL/CoPDT/DSRL:$PYTHONPATH
# export DSRL_DATASET_DIR=$HOME/.dsrl/datasets
# cd /home2/ayanroy/CBF/multi_src/9_copdt
python eval_general.py --run_dir results/copdt_general/634 \
    --eval_tasks "$TASKS" --eval_budgets "10,20,40,80" --eval_episodes 20 --project $PROJECT
