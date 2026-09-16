# Budget Adaptive CBF

This project implements a budget-conditioned offline safe reinforcement learning approach based on a control barrier function (CBF) style objective. The code trains a policy that balances reward maximization with safety constraints using a soft-constraint budget and a safety critic/value model.

The main training script is `train_offline.py`, with experiment settings in `train_config.py`. The project includes custom environments, offline dataset loading, and evaluation utilities for safety-aware RL.

## Overview

The repository is organized around:

- Training an offline agent with a reward critic, value function, and barrier/safety critic.
- Conditioning the reward actor/value on a sampled budget to generalize across soft constraint levels.
- Evaluating return and cost under different budget thresholds.
- Supporting multiple offline Safety Gym / Bullet Safety Gym / custom point-robot environments.

This is designed for research-style experiments in constrained offline RL and safety certification.

## Folder structure

- `train_offline.py` – main training entry point; loads config, builds env and dataset, trains the agent, and evaluates it.
- `train_config.py` – default configuration definitions and experiment hyperparameters.
- `cbf.sh` – example Slurm job launcher for running training on a cluster.
- `env/` – environment definitions and environment list.
  - `env/env_list.py` – supported environment IDs and names.
  - `env/point_robot.py` – custom point-robot navigation environment with hazards and goal.
- `jaxrl5/` – training code, agents, network definitions, datasets, and evaluation logic.
  - `jaxrl5/agents/cbf.py` – core CBF/budget-conditioned agent implementation.
  - `jaxrl5/data/dsrl_datasets.py` – offline dataset loader for DSRL-style datasets.
  - `jaxrl5/evaluation.py` – evaluation helpers and safety metrics.
- `launcher/` – visualization or launch utilities.

## Core idea

The agent in `jaxrl5/agents/cbf.py` combines:

- a reward actor and reward critic conditioned on a sampled budget,
- a safety critic `Q_h` and safe value `V_h` for barrier-style safety estimation,
- a cumulative cost critic/value model that tracks discounted cost under a soft budget,
- a fallback safe actor that can be used under conservative safety constraints.

The mode setting (`mode` in the config) selects the training variant. In the provided config, `mode=1` corresponds to the FISOR-style variant.

## Supported environments

The project supports a range of offline environments via `env/env_list.py`, including:

- Safety Gymnasium variants (car / point / velocity environments)
- Bullet Safety Gym environments
- PointRobot
- MetaDrive-style tasks

The environment ID is selected through the `--env_id` flag passed to `train_offline.py`.

## Install / setup

This project depends on Python packages such as:

- JAX
- Flax
- Optax
- Gymnasium
- NumPy
- Matplotlib
- ml_collections
- PyYAML
- tqdm
- wandb
- dsrl

A typical environment setup is:

```bash
pip install jax jaxlib flax optax gymnasium numpy matplotlib ml-collections pyyaml tqdm wandb dsrl
```

You may also need the dataset files used by the offline training runs to be available under the expected paths (for example, in `jaxrl5/data/` or via environment dataset loading).

## Running training

The main command is:

```bash
python train_offline.py \
  --config train_config.py:r \
  --project 140926 \
  --env_id 11 \
  --mode 1 \
  --max_steps 500001 \
  --eval_budgets "10,20,40,80"
```

Important arguments:

- `--config` selects the config object from `train_config.py` and uses the `r` variant.
- `--env_id` chooses a benchmark from `env/env_list.py`.
- `--mode` selects training mode (for example: FISOR-style mode in the provided config).
- `--max_steps` sets the number of training steps.
- `--eval_budgets` sets threshold values for evaluation-time budget-conditioned policy testing.

The project also supports a few CLI flags such as:

- `--cost_tau`
- `--reward_tau`
- `--project`

## Config file details

`train_config.py` defines a base configuration dictionary with:

- project metadata,
- environment configuration,
- dataset configuration,
- agent hyperparameters,
- evaluation budget thresholds,
- reward/cost temperature values,
- learning rates, network sizes, and training schedule settings.

The critical config section is `agent_kwargs`, which contains the CBF policy settings and budget-conditioned objective parameters such as:

- `model_cls="CBF"`
- `reward_temperature`
- `cost_temperature`
- `cost_limit`
- `cost_discount`
- `cost_expectile`
- `budget_max`
- `mode`

## Dataset handling

The offline dataset loader is implemented in `jaxrl5/data/dsrl_datasets.py`.

It supports:

- Safety Gym-style dataset files from environment `get_dataset(...)`
- custom point-robot HDF5 datasets
- cost normalization and conversion of raw per-step costs into the internal training format

This allows the same learning pipeline to run across many offline tasks.

## Evaluation

Evaluation utilities live in `jaxrl5/evaluation.py` and include:

- normalized return and cost reporting,
- conservative safety metric checks,
- barrier-based coverage and validity calculations,
- plotting support for CBF cost vs safe values.

The training loop logs metrics to Weights & Biases (`wandb`) and prints normalized return and normalized cost after evaluation intervals.

## Example Slurm run

The `cbf.sh` script demonstrates a cluster launch:

```bash
python train_offline.py --config train_config.py:r --project 140926 --env_id 11 --cost_tau 0.15 --reward_tau 0.75 --mode 1 --max_steps 500001 --eval_budgets "10,20,40,80"
```

This is useful for running larger experiments on a GPU or cluster node.

## Notes

- The project is research-oriented and expects data files to exist before training.
- It is designed around offline RL settings, not online environment interaction by default.
- The `PointRobot` environment is a custom deterministic navigation task used to evaluate safe goal-reaching under hazard constraints.

## Typical workflow

1. Choose an environment index in `env/env_list.py`.
2. Set training hyperparameters in `train_config.py`.
3. Run `train_offline.py` with the desired flags.
4. Monitor metrics logged by `wandb`.
5. Inspect evaluation results and saved model artifacts in the results directory.

## License

This repository does not include an explicit license file in the workspace snapshot. Please check the repository or project owner for licensing requirements before redistribution or publication.
