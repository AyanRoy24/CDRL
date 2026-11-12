# CBF Transfer Pipeline, v9: CoPDT / DSRL Multi-Task Pipeline

This folder is a new entry point on top of [`../8/`](../8/README.md). The COSTA multitask-ant pipeline of `../8/` (and `../7/`, `../6/`, ...) is kept as a legacy path. The default scripts now run the **DSRL task suite of CoPDT** (`CDSRL/CoPDT/OSRL/examples/train/train_mtcdt.py`) in three stages, each reading the previous stage's output. Read `../8/README.md` for the legacy pipeline, `../7/README.md` for the COSTA counterfactual context encoder, and `../4/README.md` for `IQLPooled`. This README covers only what is different from `multi_src/8`.

**Summary of the change.** Instead of COSTA maps (`ant-dir-safe` goals 0 to 2, with an `rlkit` env and COSTA discriminators), the data are the 26 offline tasks of the DSRL benchmark. Every task is pushed through a frozen per-family state/action auto-encoder into one shared latent space. A CoPDT context encoder (no discriminator, no regime labels) infers a context from a set of transitions, and one budget-adaptive `IQLPooled` policy is trained on the pooled latent data of all tasks.

## Difference from `multi_src/8`

| | `multi_src/8` | `multi_src/9_copdt` (this folder) |
|---|---|---|
| Data | COSTA multitask-ant / cheetah maps, one `hdf5` per goal | **DSRL gymnasium tasks** (`Offline*Gymnasium-v0/-v1`), 26 tasks over 13 families (`dsrl_tasks.py`) |
| Task selection | `--src_id`, `--src_goal` | **`--tasks`**: `all`, task names, family names or table indices |
| Space | raw observations and actions | **shared latent space** from frozen per-family state/action auto-encoders (`train_sa_encoder.py`) |
| Context encoder | COSTA attention encoder, distance-metric loss, regime labels, discriminator relabeling | **CoPDT encoder** (`SafetyAwareEncoder` + `MultiHeadDecoder`): no discriminator, no regime labels |
| Context input | a short segment `(s, a, c, s')` of 150 steps | **a context set** of `context_size=500` iid transitions `(s, a, s', r)`, tagged by their cost |
| Learner | `CBFContext` (plain hard-constraint CBF) | **budget-adaptive `IQLPooled`**, conditioned on `zeta` and on the budget |
| Budgets | one fixed cost limit (25) | **soft budgets `10, 20, 40, 80`** at evaluation, one network for all |
| Deployment | burn-in segment, then frozen context | **no burn-in**: `zeta` is read from the task's own offline data; the latent action is decoded to the env action |
| Stages | `train_general.py`, `adapt_*.py` | **`train_sa_encoder.py`, `train_context_encoder.py`, `train_general.py`, `eval_general.py`** |
| Config | `train_config.py` (COSTA) | **`train_config.py` rewritten**; the COSTA config is `train_config_costa.py` |
| Legacy entry point | `train_general.py` | **`train_general_costa.py`** (`cbf_costa.sh`) |

The legacy COSTA files are unchanged and still present: `train_gym.py`, `adapt_common.py`, `adapt_online.py`, `train_general_costa.py` (the old `train_general.py` with updated names), `train_config_costa.py` (identical to `multi_src/8`'s config), `cbf_context.py`, `cbf_plain.py`, `iql_pooled.py` and the rest of `models/`.

## The three stages

```
train_sa_encoder.py        stage 0   per-family state/action auto-encoders   -> results/sa_encoder/<family>/
train_context_encoder.py   stage 1   CoPDT context encoder (frozen after)    -> results/context_encoder/default/
train_general.py           stage 2   budget-adaptive IQL on the pooled latent data -> results/copdt_general/<seed>/
eval_general.py                      evaluate a saved model on chosen tasks
```

### Task suite (`dsrl_tasks.py`, new)

- `TASKS` is the 26-task table: `PointButton1/2`, `PointCircle1/2`, `PointGoal1/2`, `PointPush1/2`, `HalfCheetahVel-v0/-v1`, `HopperVel-v0/-v1`, `CarButton1/2`, `CarCircle1/2`, `CarGoal1/2`, `CarPush1/2`, `AntVel-v0/-v1`, `SwimmerVel-v0/-v1`, `Walker2dVel-v0/-v1`. The variants of one family share one auto-encoder pair, as in CoPDT's `train_sa_encoder_share.py`.
- Each task carries its state and action dimension and episode length. `select_tasks` accepts `all`, names, gym ids, family names or indices.
- `load_dsrl_task` creates the registered DSRL env, sets its target cost, and loads the pre-processed offline dataset. It aliases `safety_gymnasium.tasks.safe_velocity` to the installed `safety_velocity`, because CoPDT's DSRL fork expects the former. AntVelocity v0 and v2 exist only in CoPDT's DSRL fork (the script puts it on `PYTHONPATH`).
- `per_step_reward_scale` rescales each task so an expert's per-step reward is about 1, so one shared critic is not dominated by tasks with large rewards.

### Stage 0: state/action auto-encoders (`train_sa_encoder.py`)

For each family among `--tasks`: an action auto-encoder (only when the family's action dimension differs from `action_encode_dim=2`), then a state auto-encoder with an inverse-dynamics head on the frozen action latents. Both are JAX ports of CoPDT's `sa_encoder.py` (`models/networks/sa_encoder.py`, `models/agents/cbf/sa_encoder_trainer.py`). The checkpoint with the lowest held-out loss is kept. The frozen `Codec` moves a task between raw and latent space.

### Stage 1: CoPDT context encoder (`train_context_encoder.py`)

For every task, each update draws `meta_batch_per_task` context sets of `context_size` transitions (in the shared latent space, redrawn until a set holds a safe transition). The `SafetyAwareEncoder` maps each set to `zeta`, and a `MultiHeadDecoder` must predict, from `(zeta, s, a)`, the next-state latent, the reward and whether the step was unsafe:

```
loss = w_s * MSE(s') + w_r * MSE(r) + w_c * BCE(unsafe)      (w_s=1.0, w_r=0.0, w_c=2.0)
```

There is no discriminator and no relabeling, unlike the COSTA encoder in `multi_src/7`. Only the encoder is kept. The trainer is `copdt_context_trainer.py`; the network classes are in `models/networks/context_encoder.py`.

### Stage 2: general policy (`train_general.py`)

- **Data:** `LatentTaskData` (`copdt_common.py`) encodes every selected task once into the shared latent space and drops the raw observations. It provides the policy-transition sampler and the context-set sampler.
- **Learner:** `IQLPooled` over latent state and action spaces. The context encoder is not part of the agent here: `zeta` is computed outside, from a fresh context set per task at every update, and each transition uses its own task's `zeta`. Each update uses `2 * batch_size` transitions spread evenly over the tasks.
- **Updates:** `agent.update(batch, None, ctx, None)`, the source-only update. The budget is sampled per transition, so one network serves every budget.
- **Evaluation:** every `eval_interval`, and at the last step. For each task and each budget `C` in the eval budgets, `eval_episodes` rollouts run with a fresh context from the task's own data, the tracked budget reset to `C` and advanced by `g` after every step. The latent action is decoded to the env action by the family's action decoder (`models/eval/evaluation_dsrl.py`). Reported: `return`, `cost` (episode sum), DSRL-normalised `n_return` and `n_cost`, per task and budget, plus means over tasks.
- **Output:** `results/copdt_general/<seed>/` with `config.json` and `model<k>.pickle`.

### Evaluation script (`eval_general.py`, new)

Evaluates a saved run on chosen tasks without training, the counterpart of OSRL's `eval_mtcdt.py`. Everything (tasks, budgets, agent settings, encoder directories) is read from `<run_dir>/config.json`; flags only override. It writes `eval_<ckpt>.json` into the run directory.

```bash
python eval_general.py --run_dir results/copdt_general/634 \
  --eval_tasks "HopperVel-v0,HopperVel-v1" --eval_budgets "10,20,40,80" --eval_episodes 20
```

## Config (`train_config.py`)

Rewritten for this pipeline; the COSTA config is `train_config_costa.py`.

- **Run settings:** `project='CoPDT-DSRL'`, `eval_episodes=5` per task and budget, `eval_interval=100000`, `eval_cost_thresholds=(10, 20, 40, 80)`.
- **`dsrl_kwargs`:** `cost_limit=10` (the target cost the DSRL envs are created with) and `reward_norm=1.0`.
- **`sa_encoder_kwargs`** (stage 0): `state_encode_dim=32`, `action_encode_dim=2`, three-layer MLPs of 128, inverse-dynamics loss weight 0.1, noise `0.1`, `100` epochs of `1000` steps each for the state and action encoders.
- **`context_kwargs`** (stage 1): `context_encoding_dim=16`, `context_size=500`, `meta_batch_per_task=4`, loss weights `(1.0, 0.0, 2.0)`, `reward_scale=0.1`, `update_steps=100000`.
- **`agent_kwargs`** (stage 2): the `bcrl_det` IQL settings (`reward_tau=0.8`, `cost_expectile=0.3`, `reward_temperature=8`, `delta_max=100`, learning rates `3e-4`).

## Folder structure

- **DSRL / CoPDT pipeline:** `dsrl_tasks.py`, `copdt_common.py`, `train_sa_encoder.py`, `train_context_encoder.py`, `train_general.py`, `eval_general.py`, `train_config.py`, `cbf.sh`.
- **New models:** `models/agents/cbf/sa_encoder_trainer.py`, `models/agents/cbf/copdt_context_trainer.py`, `models/networks/sa_encoder.py`, the CoPDT classes in `models/networks/context_encoder.py`, and `models/eval/evaluation_dsrl.py`.
- **Legacy COSTA pipeline:** `train_general_costa.py`, `train_config_costa.py`, `cbf_costa.sh`, `train_gym.py`, `adapt_*.py`, and the COSTA-era agents and data code.
- **`SG/`:** a bundled `safety_gymnasium`-style package (tasks and agents).
- `results/` (`sa_encoder/`, `context_encoder/`, `copdt_general/`), `wandb/` and the Slurm logs (`antvel.txt`, `HCvel.txt`, `HopVel.txt`, `PointButton.txt`, `*_eval.txt`).

## Running

```bash
export PYTHONPATH=/home2/ayanroy/CDSRL/CoPDT/DSRL:$PYTHONPATH   # CoPDT's DSRL fork (AntVelocity v0/v2)
export DSRL_DATASET_DIR=$HOME/.dsrl/datasets                    # already-downloaded DSRL data

TASKS="HopperVel-v0,HopperVel-v1"
python train_sa_encoder.py      --config train_config.py:r --tasks "$TASKS" --project 051126
python train_context_encoder.py --config train_config.py:r --tasks "$TASKS" --project 051126 --max_steps 50000
python train_general.py         --config train_config.py:r --tasks "$TASKS" --project 051126 \
  --eval_budgets "10,20,40,80" --config.eval_episodes 10
```

The legacy COSTA run is `python train_general_costa.py --config train_config_costa.py:r --src_id 45 --src_goal "0,1,2" ...` (see `cbf_costa.sh`). On Slurm: `sbatch cbf.sh`. The active line in `cbf.sh` currently evaluates a saved run (`eval_general.py --run_dir results/copdt_general/634`) on the Hopper tasks.

Useful flags shared by the three stage scripts (`copdt_common.py`): `--tasks`, `--project`, `--wandb` (off for stdout only), `--sa_dir`, `--ctx_dir`, `--max_steps`, `--eval_budgets`.

## Notes

- The stages must run in order. Stage 1 needs the auto-encoders of every family among `--tasks`, and stage 2 needs the context encoder (it warns if the encoder was not trained on all selected tasks, and asserts that the latent dimensions agree).
- Results are not comparable with `multi_src/8`: different tasks, a latent space, a different context encoder and a budget-conditioned learner.
- The evaluation protocol has no burn-in; the context comes from offline data of the task being evaluated, not from the agent's own rollout. This is CoPDT's protocol, and it differs from the burn-in deployment of `../4/` to `../8/`.
- A full evaluation over 26 tasks and 4 budgets at 1000 steps each is expensive, hence the small `eval_episodes=5` default. The Slurm logs show runs on single families (Hopper, HalfCheetah, Walker2d, Swimmer, Ant, PointButton).
- The adapt scripts and `train_gym.py` belong to the COSTA pipeline and are not used by the DSRL scripts.
