import jax
import optax
import jax.numpy as jnp
import pickle
import os
import sys
sys.path.append('.')
import random
import numpy as np
import h5py
from absl import app, flags
import datetime
import yaml
from ml_collections import config_flags, ConfigDict
import wandb
from tqdm.auto import trange  # noqa
import gymnasium as gym
# import gym
from models.env.env_list import env_list
from models.env.point_robot import PointRobot
# CoSTA-style multitask ant envs (a shared MuJoCo Ant with a per-goal
# safe-corridor/cost function, task-selected via reset_task(goal_id)); the env
# construction + per-goal hdf5 glue lives in models/env/multitask_ant.py so the
# pooled multi-source data layer can reuse it without importing train_gym.
from models.env.multitask_ant import (
    MULTITASK_ANT_ENVS,
    make_multitask_ant_env,
    ant_goal_dataset_path,
    build_env_and_dataset,
)
from models.wrappers import wrap_gym
from models.agents import CBF, FlowTransfer, CBFTransfer, CBFPooled, IQLPooled, CBFContext, ContextEncoderTrainer
# from flow_transfer import FlowTransfer
# from cbf_transfer import CBFTransfer
from models.data.dsrl_datasets import DSRLDataset
from models.data.pooled_datasets import PooledSource, RegimeDataset
from models.eval.evaluation import evaluate, evaluate_md, evaluate_pr, evaluate_pooled #, offline_evaluation , plot_cbf_cost_vs_safe_value, calculate_coverage
import json


FLAGS = flags.FLAGS
# flags.DEFINE_string("config", "train_config.py:r", "Config file")
flags.DEFINE_integer('src_id', 23, 'Choose env')
flags.DEFINE_integer('tgt_id', 23, 'Choose env')
# flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_integer('mode', 1, 'Mode for training')
flags.DEFINE_integer('max_steps', 500_001, 'max steps')
# flags.DEFINE_integer('seed', -1, 'random seed')
flags.DEFINE_float('cost_tau', 0.1, 'cost tau')
flags.DEFINE_float('reward_tau', 0.8, 'reward tau')
# flags.DEFINE_integer('eval', 10000, 'eval steps')
flags.DEFINE_string('project', '081125', 'Name of the experiment')
flags.DEFINE_bool('transfer', False, 'Run the Ms->Mt persistent-safety transfer pipeline instead of single-domain training')
flags.DEFINE_string('eval_budgets', '', 'Comma-separated soft-constraint budgets for eval, e.g. "20,40,80"')
flags.DEFINE_list('src_goal', ['0'], 'Goal id(s) for the source env, when it is one of the multitask-ant envs (ant-dir-safe, ant-goal-safe, ant-circle-safe, ant-walk-safe); selects the reset_task(goal_id) task and the <env_name>_<goal_id>.hdf5 source dataset. A single id (e.g. "0") runs the plain single-source transfer; a comma-separated list (e.g. "0,1,3") runs the pooled multi-source / inferred-constraint-context pipeline (FPCBF_multi.tex Part III).')
flags.DEFINE_bool('eval_only', False, 'Skip training; evaluate the saved pooled model in results/<env>/<seed> (set the seed with --config.seed) on every --src_goal map (+ --tgt_goal). Requires --transfer and a multi-goal --src_goal.')
flags.DEFINE_integer('ckpt', -1, 'Checkpoint index (model<N>.pickle) for --eval_only; -1 = latest')
flags.DEFINE_integer('tgt_goal', 0, 'Goal id for the target env, when it is one of the multitask-ant envs; also used as the single-domain goal id when --transfer is not set')

config_flags.DEFINE_config_file(
    "config",
    None,
    # "configs/train_config.py",
    lock_config=False,
)


def resolve_tgt_data(details, src_goals):
    """Reference file for the target dataset. When the target goal is one of the
    source goals (same env), its <env_name>_<tgt_goal>.hdf5 buffer lives in the
    same folder as the source data, so resolve it from src_data instead of tgt_data."""
    data_kwargs = details['dataset_kwargs']
    if (details['src_env'] == details['tgt_env'] and details['tgt_env'] in MULTITASK_ANT_ENVS
            and int(details['tgt_goal']) in [int(g) for g in src_goals]):
        return data_kwargs['src_data']
    return data_kwargs['tgt_data']


def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config

# def call_main(details, env_id):
def call_main(details):
    """Direct (single-domain) CBF training. When `details['src_env']` is one of
    the multitask-ant envs, this trains and saves one converged CBF model per
    goal id listed in `details['src_goal']` -- the per-regime "direct" baseline
    used to benchmark the pooled/transfer agent against (Setting delta=0 and a
    single regime recovers this direct model, cf. `CBFPooled`'s docstring) --
    instead of the single run on the target env/goal every other dataset gets."""
    src_env = details.get('src_env')
    src_goals = details.get('src_goal')
    if src_env in MULTITASK_ANT_ENVS and src_goals:
        for goal in src_goals:
            goal = int(goal)
            print(f"[direct] training source map {src_env} goal {goal}")
            _train_direct_cbf(details, src_env, goal, group=f"{src_env}_g{goal}", tag=f"g{goal}")
        return
    _train_direct_cbf(details, details['env_name'], details['goal_id'], group=details['group'], tag=None)


def _train_direct_cbf(details, env_name, goal_id, group, tag):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    print('Training with config:', details)
    config_for_wandb = to_dict(details['agent_kwargs'])
    experiment_name = f"{details['experiment_name']}_{tag}" if tag else details['experiment_name']
    wandb.init(project=details['project'], name=experiment_name, group=group, config=config_for_wandb)
    # wandb.init( name=details['experiment_name'], group=details['group'], config=config_for_wandb)
    # details['agent_kwargs']['mode'] = wandb.config.mode
    if env_name == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(env_name)(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        # ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['pr_data'])
        ds = DSRLDataset(env, data_location=details['dataset_kwargs']['pr_data'])
    else:
        env, ds, env_max_steps = build_env_and_dataset(
            env_name, goal_id,
            details['dataset_kwargs'].get('tgt_data'),
            details['dataset_kwargs']['cost_scale'],
            details['agent_kwargs']['cost_limit'],
        )
    # ds.seed(details['dataset_kwargs']["seed"])
    ds.seed(details["seed"])
    # obs_mean = ds.obs_mean
    # obs_std = ds.obs_std
    # print('Dataset obs mean:', obs_mean, 'obs std:', obs_std)
    # Single-domain training always builds the base CBF (train_config's model_cls);
    # model_cls / cost_scale are stripped so they aren't forwarded to CBF.create.
    config_dict = dict(details['agent_kwargs'])
    config_dict.pop("model_cls", None)
    config_dict.pop("cost_scale", None)
    agent = CBF.create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )
    modeldir = f"./results/{group}/{details['seed']}"
    os.makedirs(modeldir, exist_ok=True)
    save_time, eval_num = 1, 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=experiment_name):
        sample = ds.sample_jax(details['batch_size'])
        agent, info = agent.update(sample)
        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)
        # if i >= (details['max_steps']-20):
        if i % details['eval_interval'] == 0:
            if i == details['max_steps'] - 1:
                agent.save(modeldir, save_time)
                save_time += 1
            if env_name == 'PointRobot':
                eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            # # elif env_id >= 30:
            # #     eval_info = evaluate_md(obs_mean, obs_std, details['seed'], env_id, eval_num, agent, env, details['eval_episodes'], render=False) #, save_video=True, )
            # #         # eval_num += 1
            #         # eval_info = evaluate(agent, env, details['eval_episodes'], save_video=True, render=True)
            # else:
        eval_info = evaluate(details['seed'], agent, env, details['eval_episodes'],
                         cost_thresholds=tuple(details['eval_cost_thresholds']))

        if env_name != 'PointRobot' : #and 'n_return' not in eval_info:
            eval_info["n_return"], eval_info["n_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])

        # print ({f"eval/{k}": v for k, v in eval_info.items()})
        wandb.log({f"{k}": v for k, v in eval_info.items()} , step=i)
        # wandb.log({f"{k}": v for k, v in eval_info.items()} )
    # converged model is guaranteed saved even if max_steps-1 wasn't hit exactly
    # on an eval_interval boundary (e.g. max_steps not a multiple of it).
    if (details['max_steps'] - 1) % details['eval_interval'] != 0:
        agent.save(modeldir, save_time)
    wandb.finish()


# def call_main_transfer(details, env_id):
def call_main_transfer(details):
    """Runs the full Ms -> Mt pipeline from the transfer formulation:
    Step 1 (encoder/decoder reconstruction) -> Step 2 (OT-coupled flow matching + safety
    alignment) -> Step 3 (barrier trained to agree at z and at its forward projection
    f_eta(z), reward critic/policy trained on source only) -> Step 4 (eval on target,
    filtering pi_theta's actions with the already-target-valid Q_h^phi).
    """
    # assert details['env_name'] == 'PointRobot', "The transfer pipeline currently assumes a PointRobot-style CMDP pair (Ms, Mt)."
    src_goals = details['src_goal'] if isinstance(details['src_goal'], (list, tuple)) else [details['src_goal']]
    src_goals = [int(g) for g in src_goals]
    if len(src_goals) > 1:
        # A list of source goal ids -> pool the per-goal buffers, infer each
        # trajectory's constraint regime from its cost signal, and train a
        # task- and budget-conditioned source CBF before it transfers to the
        # target (FPCBF_multi.tex Part III).
        return call_main_transfer_pooled(details, src_goals)
    src_goal = src_goals[0]
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    print('Training (transfer) with config:', details)
    config_for_wandb = to_dict(details['agent_kwargs'])
    config_for_wandb.update(to_dict(details['transfer_kwargs']))
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=config_for_wandb)

    # Each phase below (recon / regressor / flow / main training) is its own loop whose `i`
    # restarts at 0, but wandb enforces one monotonically-increasing step per run. Giving each
    # phase its own step metric avoids "Tried to log to step X that is less than the current
    # step" warnings instead of fighting wandb's global step counter. See wandb.me/define-metric.
    wandb.define_metric("transfer/recon/step")
    wandb.define_metric("transfer/recon/*", step_metric="transfer/recon/step")
    wandb.define_metric("transfer/reg/step")
    wandb.define_metric("transfer/reg/*", step_metric="transfer/reg/step")
    wandb.define_metric("transfer/flow/step")
    wandb.define_metric("transfer/flow/*", step_metric="transfer/flow/step")
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")

    data_kwargs = details['dataset_kwargs']

    if details['src_env'] == 'PointRobot':
        src_env = PointRobot(id=0, seed=0)
        tgt_env = PointRobot(
            id=0, seed=0,
            hazard_position_list=data_kwargs['target_hazard_position_list'],
            hazard_size=data_kwargs['target_hazard_size'],
        )
        print ('Source env hazard positions:', src_env.hazard_position_list, 'hazard size:', src_env.hazard_size)
        print ('Target env hazard positions:', tgt_env.hazard_position_list, 'hazard size:', tgt_env.hazard_size)

        assert data_kwargs['src_data'] is not None, "No source data (Ds) for Point Robot"
        assert data_kwargs['tgt_data'] is not None, "No target data (Dt) for Point Robot"
        ds_source = DSRLDataset(src_env, data_location=data_kwargs['src_data'])
        ds_target = DSRLDataset(tgt_env, data_location=data_kwargs['tgt_data'])

    else:
        src_env, ds_source, src_env_max_steps = build_env_and_dataset(
            details['src_env'], src_goal, data_kwargs['src_data'],
            details['dataset_kwargs']['cost_scale'], details['agent_kwargs']['cost_limit'],
        )
        tgt_env, ds_target, tgt_env_max_steps = build_env_and_dataset(
            details['tgt_env'], details['tgt_goal'], resolve_tgt_data(details, src_goals),
            details['dataset_kwargs']['cost_scale'], details['agent_kwargs']['cost_limit'],
        )

    ds_source.seed(details['seed'])
    ds_target.seed(details['seed'] + 1)

    transfer_kwargs = to_dict(details['transfer_kwargs'])
    schedule = to_dict(details['transfer_schedule'])
    ot_bs = schedule['ot_batch_size']

    flow_transfer = FlowTransfer.create(details['seed'], src_env.observation_space, src_env.action_space, **transfer_kwargs)

    # Step 1: reconstruction on Ds u Dt, fixing the shared coordinate system.
    # (sample_jax always pulls every field - its jitted sampler is cached on first
    # call per dataset, so a `keys=` subset here would silently stick for later calls.)
    for i in trange(schedule['recon_steps'], desc='transfer: encoder/decoder'):
        src = ds_source.sample_jax(ot_bs)
        tgt = ds_target.sample_jax(ot_bs)
        merged = {k: jnp.concatenate([src[k], tgt[k]], axis=0) for k in ('observations', 'actions')}
        flow_transfer, info = flow_transfer.update_reconstruction(merged)
        if i % details['log_interval'] == 0:
            log_info = {f"transfer/recon/{k}": v for k, v in info.items()}
            log_info["transfer/recon/step"] = i
            wandb.log(log_info)
# save the autoencoder after reconstruction training, so that it can be reloaded for Step 2.
# save the flow_transfer object after reconstruction training, so that it can be reloaded for Step 2.

    # Step 2(g): pretrain h_hat_t^xi on Dt, then freeze it for the alignment loss.
    for i in trange(schedule['reg_steps'], desc='transfer: safety regressor'):
        tgt = ds_target.sample_jax(ot_bs)
        flow_transfer, info = flow_transfer.pretrain_safety_regressor(tgt)
        if i % details['log_interval'] == 0:
            log_info = {f"transfer/reg/{k}": v for k, v in info.items()}
            log_info["transfer/reg/step"] = i
            wandb.log(log_info)
# save the regressor after pretraining, so that it can be reloaded for Step 2.

    # Step 2(a)-(h): OT-coupled flow matching + safety alignment, fitting v_eta.
    for i in trange(schedule['flow_train_steps'], desc='transfer: flow map'):
        src = ds_source.sample_jax(ot_bs)
        tgt = ds_target.sample_jax(ot_bs)
        flow_transfer, info = flow_transfer.update_flow(src, tgt)
        if i % details['log_interval'] == 0:
            log_info = {f"transfer/flow/{k}": v for k, v in info.items()}
            log_info["transfer/flow/step"] = i
            wandb.log(log_info)
# save the flow_transfer object after flow training, so that it can be reloaded for Step 3.

    # Step 3: safety critic trained to simultaneously agree on Zs and on the forward-projected Zt;
    # reward critic/value/policy trained exactly as CBF, on source data only.    
    config_dict = dict(details['agent_kwargs'])
    config_dict.pop("model_cls")
    config_dict.pop("cost_scale")
    agent = CBFTransfer.create(
        details['seed'], src_env.observation_space, src_env.action_space,
        autoencoder=flow_transfer.autoencoder,
        velocity=flow_transfer.velocity,
        latent_dim=transfer_kwargs['latent_dim'],
        flow_steps=transfer_kwargs['flow_steps'],
        **config_dict,
    )

    modeldir = f"./results/{details['group']}/{details['seed']}"
    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        src_sample = ds_source.sample_jax(details['batch_size'])
                # Dt is small and sparse relative to Ds, so it is cycled through (stratified,
                # without replacement) each epoch rather than sampled i.i.d. with replacement.
                # tgt_sample = ds_target.sample_cycled(details['batch_size'])
        tgt_sample = ds_target.sample_jax(details['batch_size'])
        agent, info = agent.update(src_sample, tgt_sample)
        # this update() call is the same as in call_main, but the agent is now a CBFTransfer with frozen encoder and flow map.

        if i % details['log_interval'] == 0:
            log_info = {f"train/{k}": v for k, v in info.items()}
            log_info["train/step"] = i
            wandb.log(log_info)
        if i == details['max_steps'] - 1 and details['src_env'] == 'PointRobot':
            # as done in call_main: checkpoint the converged models at each eval step.
            # flow_transfer bundles the encoder/decoder (autoencoder), flow model (velocity)
            # and regressor (safety_regressor); agent is the CBFTransfer model (reward critic/
            # value/actor trained on source, safety critic valid on both source and target).
            print(f"Saving models at step {i} to {modeldir}...")
            flow_transfer.save(modeldir, save_time)
            agent.save(modeldir, save_time)
            save_time += 1

    # Step 4: eval on Mt. No projection at deployment - Q_h^phi is already valid on target latents.
    
        # if i >= (details['max_steps']-20) and details['src_env'] != 'PointRobot':
        if i % details['eval_interval'] == 0:
            if details['src_env'] == 'PointRobot':
                eval_info = evaluate_pr(agent, tgt_env, details['eval_episodes'])
            else:
                eval_info = evaluate(details['seed'], agent, tgt_env, details['eval_episodes'],
                                     cost_thresholds=tuple(details['eval_cost_thresholds']))
                if 'n_return' not in eval_info:
                    eval_info["n_return"], eval_info["n_cost"] = tgt_env.get_normalized_score(eval_info["return"], eval_info["cost"])
            print ({f"eval/n_ret": eval_info["n_return"], f"eval/n_cost": eval_info["n_cost"]})
            wandb.log({f"{k}": v for k, v in eval_info.items()} , step=i) 

    # eval_info = evaluate_pr(agent, env_target, details['eval_episodes'])
    # wandb.log({f"{k}": v for k, v in eval_info.items()})
        # if i % details['eval_interval'] == 0 and details['src_env'] == 'PointRobot':
        #     eval_info = evaluate_pr(agent, tgt_env, details['eval_episodes'])
        #     print ({f"eval/{k} for step {i}": v for k, v in eval_info.items()})
        #     wandb.log({f"{k}": v for k, v in eval_info.items()}, step=i)


def _make_iql_pooled(seed, obs_space, act_space, context_encoder, pk, agent_kwargs, m):
    if agent_kwargs.get('model_cls') == 'CBFContext':
        # plain hard-constraint CBF as the learner: no budget, context-conditioned via the widened obs
        kw = dict(agent_kwargs)
        kw.pop('model_cls', None); kw.pop('cost_scale', None)
        return CBFContext.create(seed, obs_space, act_space, context_encoder=context_encoder,
                                 context_dim=pk['context_dim'], seg_m=m, **kw)
    return IQLPooled.create(
        seed, obs_space, act_space,
        context_encoder=context_encoder,
        context_dim=pk['context_dim'],
        seg_m=m,
        actor_lr=agent_kwargs['actor_lr'],
        critic_lr=agent_kwargs['critic_lr'],
        value_lr=agent_kwargs['value_lr'],
        cost_critic_lr=agent_kwargs['cbf_lr'],
        cost_value_lr=agent_kwargs['cbf_lr'],
        discount=agent_kwargs['discount'],
        cost_discount=agent_kwargs['cost_discount'],
        expectile=agent_kwargs['reward_tau'],
        cost_expectile=agent_kwargs['cost_expectile'],
        beta=agent_kwargs['reward_temperature'],
        delta_max=pk['delta_max'],
        actor_weight_decay=agent_kwargs['actor_weight_decay'],
        decay_steps=agent_kwargs['decay_steps'],
        value_layer_norm=agent_kwargs['value_layer_norm'],
    )


def evaluate_only_pooled(details, src_goals):
    """Evaluate a saved pooled model (results/<group>/<seed>/model<N>.pickle +
    context_encoder<N>.pickle) on every source-goal map (plus the target goal if
    it is not one of them). No training, no datasets loaded. The context is
    re-inferred per episode by the burn-in inside evaluate_pooled."""
    import re
    pk = to_dict(details['pooled_kwargs'])
    seed = details['seed']
    modeldir = f"./results/{details['group']}/{seed}"
    if FLAGS.ckpt >= 0:
        ckpt = FLAGS.ckpt
    else:
        found = [int(re.findall(r'model(\d+)\.pickle', f)[0]) for f in os.listdir(modeldir) if re.fullmatch(r'model\d+\.pickle', f)]
        assert found, f"no model*.pickle in {modeldir}"
        ckpt = max(found)
    print(f"[eval-only] loading checkpoint {ckpt} from {modeldir}")

    goals = list(src_goals)
    tgt_goal = int(details['tgt_goal'])
    if tgt_goal not in goals:
        goals.append(tgt_goal)
    cost_limit = details['agent_kwargs']['cost_limit']
    envs = {g: wrap_gym(make_multitask_ant_env(details['tgt_env'], g), cost_limit=cost_limit) for g in goals}
    obs_space, act_space = envs[goals[0]].observation_space, envs[goals[0]].action_space
    m = pk['seg_len']

    ctx_trainer = ContextEncoderTrainer.create(
        seed, obs_space.shape[0], act_space.shape[0], m,
        context_dim=pk['context_dim'], hidden_dims=tuple(pk['ctx_hidden_dims']),
        decoder_hidden_dims=tuple(pk['decoder_hidden_dims']),
        lr=pk['ctx_lr'], decoder_weight=pk['decoder_weight'],
    ).load(os.path.join(modeldir, f"context_encoder{ckpt}.pickle"))
    agent = _make_iql_pooled(seed, obs_space, act_space, ctx_trainer.encoder, pk, details['agent_kwargs'], m)
    agent = agent.load(os.path.join(modeldir, f"model{ckpt}.pickle"))

    results = {}
    for g in goals:
        info = evaluate_pooled(
            seed, agent, envs[g], details['eval_episodes'],
            seg_len=m, context_dim=pk['context_dim'],
            cost_thresholds=tuple(details['eval_cost_thresholds']),
        )
        results.update({f"goal_{g}/{k}": v for k, v in info.items()})
        print(f"[eval-only] goal {g}: return={info['return']:.1f} cost(rate)={info['cost']:.4f} "
              f"cost_sum={info['cost_sum']:.2f} n_return={info.get('n_return', float('nan')):.1f}")
    for k in ("return", "cost", "cost_sum", "n_return", "n_cost"):
        vals = [results[f"goal_{g}/{k}"] for g in goals if f"goal_{g}/{k}" in results]
        if vals:
            results[f"mean/{k}"] = float(np.mean(vals))
    print("[eval-only] mean over goals:", {k: v for k, v in results.items() if k.startswith("mean/")})
    out = os.path.join(modeldir, f"eval_only_ckpt{ckpt}.json")
    with open(out, "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=4)
    print(f"[eval-only] saved {out}")


def _evaluate_on_goals(details, agent, envs, seg_len, context_dim):
    """evaluate_pooled on each goal's env: `goal_<g>/<metric>` per goal, plus the
    headline metrics (`return`, `cost`, ...) averaged over the goals."""
    per_goal = {
        g: evaluate_pooled(details['seed'], agent, env, details['eval_episodes'],
                           seg_len=seg_len, context_dim=context_dim,
                           cost_thresholds=tuple(details['eval_cost_thresholds']))
        for g, env in envs.items()
    }
    results = {f"goal_{g}/{k}": v for g, info in per_goal.items() for k, v in info.items()}
    for k in next(iter(per_goal.values())):
        if '/' not in k:
            results[k] = float(np.mean([info[k] for info in per_goal.values()]))
    return results


def call_main_transfer_pooled(details, src_goals, eval_callback=None, save=True):
    """FPCBF_multi.tex Part III -- Pooled Multi-Source Training with Inferred
    Constraint Context, entered from call_main_transfer when --src_goal lists
    more than one multitask-ant goal id.  Three stages:

      1. context encoder q_xi  -- CDSRL/COSTA/COSTA's own objective (attention-
                                   pooled encoder + distance-metric loss + a
                                   decoder reconstructing the per-step cost bit,
                                   `ContextEncoderTrainer`/`OfflineMetaLearner`),
                                   on cost-annotated segments
      2. bcrl_det.IQL           -- CBF/bcrl's budget-conditioned cost/reward
                                   critics + dual (budget-conditioned / safe-
                                   fallback) actor (`IQLPooled`), context-
                                   conditioned on zeta and trained on the pooled
                                   D_s u D_t minibatch -- replaces CBFPooled's
                                   budget-adaptive-CBF bookkeeping (no hard
                                   barrier Q_h, no latent autoencoder/OT flow:
                                   everything operates on raw observations)
      3. deployment             -- burn-in freeze of zeta_hat, budget recursion,
                                   the dual actor's own budget/safe-actor switch

    Each goal id is one constraint regime c^(k) (the per-goal safe corridor of
    ant_dir_safe.py &co.), loaded from its <env_name>_<goal_id>.hdf5 buffer.  The
    task-inference module (q_xi) reads the regime off each trajectory's cost
    signal so the shared driving skill is pooled while the disagreeing cost
    labels are kept apart -- instead of averaging a safe state in one regime
    against an unsafe state in another (Prop. pool-unsound).

    Hyperparameter-tuning hooks (tune_pooled.py): `eval_callback(step, eval_info)` is
    called after every evaluation (it may raise to prune the run); `save=False`
    skips writing checkpoints.  Returns the last `eval_info`.

    `details['tgt_goal'] = None` (train_general.py) trains the general policy: no
    target-side terms at all, evaluated on every source goal.  Otherwise the
    target-side terms are kept, as the pooled-only variant.
    """
    assert details['src_env'] in MULTITASK_ANT_ENVS and details['tgt_env'] in MULTITASK_ANT_ENVS, \
        "pooled multi-source (--src_goal list) is only defined for the multitask-ant envs"

    pk = to_dict(details['pooled_kwargs'])
    print('Training (pooled, bcrl_det IQL) with config:', details)

    config_for_wandb = {**to_dict(details['agent_kwargs']), **pk}
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=config_for_wandb)
    for phase in ("ctx", "train"):
        wandb.define_metric(f"pooled/{phase}/step")
        wandb.define_metric(f"pooled/{phase}/*", step_metric=f"pooled/{phase}/step")

    data_kwargs = details['dataset_kwargs']
    seed = details['seed']
    cost_scale = details['dataset_kwargs']['cost_scale']
    cost_limit = details['agent_kwargs']['cost_limit']

    # ---- datasets: K pooled source regimes (+ a distinct target buffer) --------
    # The target is a genuinely separate domain only when its env differs from the
    # source env, or its goal id is not already in the pooled source list.  When it
    # is one of the pooled regimes there is no separate D_t: that regime is reused
    # for every target-side role.
    general = details.get('tgt_goal') is None

    pooled = PooledSource(details['src_env'], src_goals, data_kwargs['src_data'],
                          cost_scale=cost_scale, seed=seed)
    relabel = bool(pk.get('ctx_relabel', False))
    if relabel:   # COSTA's cost discriminators relabel every regime's transitions for the encoder's counterfactual loss
        pooled.attach_relabelers(pk['costa_dir'], pk.get('relabel_device', 'cpu'))
    if general:
        ds_target = None
        eval_envs = {g: wrap_gym(make_multitask_ant_env(details['src_env'], g), cost_limit=cost_limit)
                     for g in src_goals}
        obs_space, act_space = eval_envs[src_goals[0]].observation_space, eval_envs[src_goals[0]].action_space
    else:
        tgt_goal = int(details['tgt_goal'])
        distinct_target = (details['src_env'] != details['tgt_env']) or (tgt_goal not in src_goals)
        tgt_data = resolve_tgt_data(details, src_goals)
        tgt_env, _, _ = build_env_and_dataset(details['tgt_env'], tgt_goal,
                                              tgt_data, cost_scale, cost_limit)
        if distinct_target:
            ds_target = RegimeDataset(
                make_multitask_ant_env(details['tgt_env'], tgt_goal),
                regime_id=-1, goal_id=tgt_goal,
                data_location=ant_goal_dataset_path(details['tgt_env'], tgt_goal, tgt_data),
                cost_scale=cost_scale,
            )
            ds_target.seed(seed + 1)
        else:
            ds_target = pooled.regimes[src_goals.index(tgt_goal)]
            print(f"[pooled] target goal {tgt_goal} is pooled source regime "
                  f"{src_goals.index(tgt_goal)} -> no separate D_t; reusing that regime "
                  f"for the target-side terms")
        obs_space, act_space = tgt_env.observation_space, tgt_env.action_space
    m = pk['seg_len']
    obs_dim, action_dim = obs_space.shape[0], act_space.shape[0]

    # ===================== Stage 1: constraint-context encoder q_xi =============
    ctx_trainer = ContextEncoderTrainer.create(
        seed, obs_dim, action_dim, m,
        context_dim=pk['context_dim'], hidden_dims=tuple(pk['ctx_hidden_dims']),
        decoder_hidden_dims=tuple(pk['decoder_hidden_dims']),
        lr=pk['ctx_lr'], decoder_weight=pk['decoder_weight'],
        contrastive_weight=pk.get('ctx_contrastive_weight', 0.1),
    )
    for i in trange(pk['ctx_steps'], desc='pooled: context encoder'):
        if relabel:
            segs, labels, pos, neg = pooled.sample_contrastive(pk['ctx_batch_per_regime'], m, pk['cost_anchor_frac'],
                                                               counterfactual=True)
            ctx_trainer, info = ctx_trainer.update_cf(jnp.asarray(segs), jnp.asarray(labels),
                                                      jnp.asarray(pos), jnp.asarray(neg))
        else:
            segs, labels = pooled.sample_contrastive(pk['ctx_batch_per_regime'], m, pk['cost_anchor_frac'])
            ctx_trainer, info = ctx_trainer.update(jnp.asarray(segs), jnp.asarray(labels))
        if i % details['log_interval'] == 0:
            wandb.log({**{f"pooled/ctx/{k}": v for k, v in info.items()}, "pooled/ctx/step": i})

    def _embed_segments(segments):
        return np.asarray(ctx_trainer.embed(jnp.asarray(segments)))

    if not general:
        tgt_segs, _ = ds_target.sample_segments(512, m, ds_target.np_random, pk['cost_anchor_frac'], return_transitions=True)
        tgt_seg_emb = _embed_segments(tgt_segs)
        ctx_t = tgt_seg_emb.mean(0)
        ctx_t_dispersion = float(np.linalg.norm(tgt_seg_emb - ctx_t, axis=1).mean())
        print(f"[pooled] zeta_t dispersion={ctx_t_dispersion:.4f}")
        wandb.log({"pooled/ctx/zeta_t_dispersion": ctx_t_dispersion})
        ctx_t = jnp.asarray(ctx_t, dtype=jnp.float32)

    # ===================== Stage 2: bcrl_det.IQL, context-conditioned ==========
    agent = _make_iql_pooled(seed, obs_space, act_space, ctx_trainer.encoder, pk, details['agent_kwargs'], m)
    if not general:
        agent = agent.set_context(ctx_t)

    modeldir = f"./results/{details['group']}/{seed}"
    if save:
        os.makedirs(modeldir, exist_ok=True)
    save_time = 1
    last_eval = None
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        # the general policy has no target half, so its source half doubles to keep the same update size
        src_size = 2 * details['batch_size'] if general else details['batch_size']
        segs, src_batch, labels = pooled.sample_pooled_transitions(src_size, m, pk['cost_anchor_frac'])
        ctx_s = ctx_trainer.embed(jnp.asarray(segs))                       # detached: q_xi is frozen
        src_batch = {k: jnp.asarray(v) for k, v in src_batch.items()}
        if general:
            agent, info = agent.update(src_batch, None, ctx_s, None)
        else:
            _, tgt_np = ds_target.sample_segments(details['batch_size'], m, ds_target.np_random,
                                                  pk['cost_anchor_frac'], return_transitions=True)
            tgt_batch = {k: jnp.asarray(v) for k, v in tgt_np.items()}
            agent, info = agent.update(src_batch, tgt_batch, ctx_s, ctx_t)

        if i % details['log_interval'] == 0:
            wandb.log({**{f"pooled/train/{k}": v for k, v in info.items()}, "pooled/train/step": i})

        if i % details['eval_interval'] == 0 and (i > 0 or eval_callback is None):
            if general:
                eval_info = _evaluate_on_goals(details, agent, eval_envs, m, pk['context_dim'])
            else:
                eval_info = evaluate_pooled(
                    details['seed'], agent, tgt_env, details['eval_episodes'],
                    seg_len=m, context_dim=pk['context_dim'],
                    cost_thresholds=tuple(details['eval_cost_thresholds']),
                )
            print({f"eval/{k}": v for k, v in eval_info.items()})
            wandb.log({f"{k}": v for k, v in eval_info.items()}, step=i)
            last_eval = eval_info
            if eval_callback is not None:
                eval_callback(i, eval_info)
        if save and i == details['max_steps'] - 1:
            ctx_trainer.save(modeldir, save_time)
            agent.save(modeldir, save_time)
            save_time += 1
    return last_eval
                        

def main(_):
    parameters = FLAGS.config
    src_id = FLAGS.src_id
    tgt_id = FLAGS.tgt_id
    mode = FLAGS.mode
    # algo = 'fisor' if mode == 1 else 'tanh'
    parameters['agent_kwargs']['mode'] = mode
    parameters['agent_kwargs']['cost_tau'] = FLAGS.cost_tau
    parameters['agent_kwargs']['reward_tau'] = FLAGS.reward_tau
    if FLAGS.eval_budgets:
        parameters['eval_cost_thresholds'] = tuple(
            float(x) for x in FLAGS.eval_budgets.split(',') if x.strip()
        )
    algo = {1: 'fisor', 2: 'yvbf', 3: 'rcrl', 4:'leak'}.get(mode, 'tanh')

    # algo = 'tanh'
    parameters['project'] = FLAGS.project
    parameters['src_env'] = env_list[src_id]
    parameters['tgt_env'] = env_list[tgt_id]
    src_goals = [int(g) for g in FLAGS.src_goal]
    parameters['src_goal'] = src_goals
    parameters['tgt_goal'] = FLAGS.tgt_goal
    pooled = FLAGS.transfer and len(src_goals) > 1
    # Single-domain (non-transfer) training runs on the target env/goal --
    # *unless* src_env is one of the multitask-ant envs, in which case call_main
    # ignores tgt_env/tgt_goal entirely and instead trains one direct CBF model
    # per --src_goal id on src_env (see call_main's docstring). --tgt_id/--tgt_goal
    # default to placeholders (env_list[23], goal 0) that are never even read in
    # that case, so naming/logging must key off src_env here too, or the wandb
    # run and results/ path end up stamped with that irrelevant default target.
    direct_multi_source = (not FLAGS.transfer) and parameters['src_env'] in MULTITASK_ANT_ENVS and len(src_goals) > 0
    parameters['env_name'] = parameters['tgt_env']
    parameters['goal_id'] = FLAGS.tgt_goal

    parameters['group'] = parameters['src_env'] if direct_multi_source else parameters['tgt_env']
    # parameters['experiment_name'] = str(env_id) + '_'  + str(parameters['env_name']) + '_' + str(parameters['dataset_kwargs']['seed']) #str(np.random.randint(1000))
    _tag = '_pooled' if pooled else ('_transfer' if FLAGS.transfer else '')
    if direct_multi_source:
        parameters['experiment_name'] = str(src_id) + '_direct_' + str(parameters['src_env']) + '_' + str(parameters['seed'])
    else:
        parameters['experiment_name'] = str(tgt_id) + _tag + '_' + str(parameters['tgt_env']) + '_' + str(parameters['seed']) #str(np.random.randint(1000))

    if parameters['src_env'] == 'PointRobot':
        parameters['max_steps'] = 100001
        parameters['eval_episodes'] = 20
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        # parameters['eval_episodes'] = 2
        # parameters['agent_kwargs']['cost_temperature'] = 2
        parameters['agent_kwargs']['reward_temperature'] = 5
        # parameters['agent_kwargs']['cost_tau'] = 0.01
        parameters['agent_kwargs']['cost_ub'] = 150
        parameters['agent_kwargs']['N'] = 8
    
    if src_id >= 23 and src_id<=40:  # Bullet safety gym envs
        parameters['agent_kwargs']['cost_limit'] = 5

    if FLAGS.transfer:
        # so config.json records which class the checkpoint actually is - viz_map.py and any
        # other loader dispatches on this field to know whether to rebuild a CBF, a CBFTransfer
        # or a CBFPooled skeleton.
        parameters['agent_kwargs']['model_cls'] = 'IQLPooled' if pooled else 'CBFTransfer'

    # print(parameters)

    if FLAGS.eval_only:
        assert pooled, "--eval_only is implemented for the pooled pipeline (--transfer with a multi-goal --src_goal)"
        assert parameters['seed'] >= 0 and os.path.isdir(f"./results/{parameters['tgt_env']}/{parameters['seed']}"), \
            "pass the training seed via --config.seed=<seed>"
        return evaluate_only_pooled(parameters, src_goals)

    # For direct_multi_source this is just a manifest of the whole sweep --
    # call_main saves each goal's own converged model under its own
    # results/<src_env>_g<goal>/<seed>/ dir (see call_main's docstring).
    manifest_group = parameters['src_env'] if direct_multi_source else parameters['tgt_env']
    if not os.path.exists(f"./results/{manifest_group}/{parameters['seed']}"):
        os.makedirs(f"./results/{manifest_group}/{parameters['seed']}")
    with open(f"./results/{manifest_group}/{parameters['seed']}/config.json", "w") as f:
        json.dump(to_dict(parameters), f, indent=4)
    
    # call_main(parameters, parameters['dataset_kwargs']['env_id'])
    if FLAGS.transfer:
        print('Running transfer pipeline...')
        call_main_transfer(parameters)
    else:
        call_main(parameters)
    # call_main(parameters)

if __name__ == '__main__':
    app.run(main)
