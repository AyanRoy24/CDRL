import jax
import optax
import jax.numpy as jnp
import pickle
import os
import sys
sys.path.append('.')
import random
import numpy as np
from absl import app, flags
import datetime
import yaml
from ml_collections import config_flags, ConfigDict
import wandb
from tqdm.auto import trange  # noqa
import gymnasium as gym
# import gym
from env_list import env_list
from point_robot import PointRobot
from wrappers import wrap_gym
from cbf import CBF
from flow_transfer import FlowTransfer
from cbf_transfer import CBFTransfer
from dsrl_datasets import DSRLDataset
from evaluation import evaluate, evaluate_md, evaluate_pr #, offline_evaluation , plot_cbf_cost_vs_safe_value, calculate_coverage
import json

FLAGS = flags.FLAGS
# flags.DEFINE_string("config", "train_config.py:r", "Config file")
flags.DEFINE_integer('env_id', 23, 'Choose env')
# flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_integer('mode', 1, 'Mode for training')
# flags.DEFINE_integer('max_steps', 500_001, 'max steps')
# flags.DEFINE_integer('eval', 10000, 'eval steps')
flags.DEFINE_string('project', '081125', 'Name of the experiment')
flags.DEFINE_bool('transfer', False, 'Run the Ms->Mt persistent-safety transfer pipeline instead of single-domain training')

config_flags.DEFINE_config_file(
    "config",
    None,
    # "configs/train_config.py",
    lock_config=False,
)


def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config

def call_main(details, env_id):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    print('Training with config:', details)
    config_for_wandb = to_dict(details['agent_kwargs'])
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=config_for_wandb)
    # wandb.init( name=details['experiment_name'], group=details['group'], config=config_for_wandb)
    # details['agent_kwargs']['mode'] = wandb.config.mode
    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        # ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['pr_data'])
        ds = DSRLDataset(env, data_location=details['dataset_kwargs']['pr_data'])
    else:
        env = gym.make(details['env_name']) #,use_render=True)
        # ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        ds = DSRLDataset(env, cost_scale=details['dataset_kwargs']['cost_scale'])#, ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    # ds.seed(details['dataset_kwargs']["seed"])
    ds.seed(details["seed"])
    # obs_mean = ds.obs_mean
    # obs_std = ds.obs_std
    # print('Dataset obs mean:', obs_mean, 'obs std:', obs_std)
    config_dict = dict(details['agent_kwargs'])
    model_cls = config_dict.pop("model_cls") 
    config_dict.pop("cost_scale") 
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )
    save_time, eval_num = 1, 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])     
        agent, info = agent.update(sample)
        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)
        # if i >= (details['max_steps']-20):
        if i % details['eval_interval'] == 0:
            agent.save(f"./results/{details['group']}/{details['seed']}", save_time)
            save_time += 1
            if details['env_name'] == 'PointRobot':
                eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            # # elif env_id >= 30:
            # #     eval_info = evaluate_md(obs_mean, obs_std, details['seed'], env_id, eval_num, agent, env, details['eval_episodes'], render=False) #, save_video=True, )
            # #         # eval_num += 1        
            #         # eval_info = evaluate(agent, env, details['eval_episodes'], save_video=True, render=True)
            # else:
    eval_info = evaluate(details['seed'], agent, env, details['eval_episodes']) #, details['agent_kwargs']['cost_limit'])

    if details['env_name'] != 'PointRobot':
        eval_info["n_return"], eval_info["n_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
    
    # print ({f"eval/{k}": v for k, v in eval_info.items()})
    wandb.log({f"{k}": v for k, v in eval_info.items()} , step=i)
    # wandb.log({f"{k}": v for k, v in eval_info.items()} )


def call_main_transfer(details, env_id):
    """Runs the full Ms -> Mt pipeline from the transfer formulation:
    Step 1 (encoder/decoder reconstruction) -> Step 2 (OT-coupled flow matching + safety
    alignment) -> Step 3 (barrier trained to agree at z and at its forward projection
    f_eta(z), reward critic/policy trained on source only) -> Step 4 (eval on target,
    filtering pi_theta's actions with the already-target-valid Q_h^phi).
    """
    assert details['env_name'] == 'PointRobot', "The transfer pipeline currently assumes a PointRobot-style CMDP pair (Ms, Mt)."
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
    env = PointRobot(id=0, seed=0)
    env_target = PointRobot(
        id=0, seed=0,
        hazard_position_list=data_kwargs['target_hazard_position_list'],
        hazard_size=data_kwargs['target_hazard_size'],
    )
    print ('Source env hazard positions:', env.hazard_position_list, 'hazard size:', env.hazard_size)
    print ('Target env hazard positions:', env_target.hazard_position_list, 'hazard size:', env_target.hazard_size)

    assert data_kwargs['pr_data'] is not None, "No source data (Ds) for Point Robot"
    assert data_kwargs['pr_data_target'] is not None, "No target data (Dt) for Point Robot"
    ds_source = DSRLDataset(env, data_location=data_kwargs['pr_data'])
    ds_target = DSRLDataset(env_target, data_location=data_kwargs['pr_data_target'])
    ds_source.seed(details['seed'])
    ds_target.seed(details['seed'] + 1)

    transfer_kwargs = to_dict(details['transfer_kwargs'])
    schedule = to_dict(details['transfer_schedule'])
    ot_bs = schedule['ot_batch_size']

    flow_transfer = FlowTransfer.create(details['seed'], env.observation_space, env.action_space, **transfer_kwargs)

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
        details['seed'], env.observation_space, env.action_space,
        autoencoder=flow_transfer.autoencoder,
        velocity=flow_transfer.velocity,
        latent_dim=transfer_kwargs['latent_dim'],
        flow_steps=transfer_kwargs['flow_steps'],
        **config_dict,
    )

    modeldir = f"./results/{details['group']}/{details['seed']}"
    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds_source.sample_jax(details['batch_size'])
        agent, info = agent.update(sample)
        # this update() call is the same as in call_main, but the agent is now a CBFTransfer with frozen encoder and flow map.

        if i % details['log_interval'] == 0:
            log_info = {f"train/{k}": v for k, v in info.items()}
            log_info["train/step"] = i
            wandb.log(log_info)
        # if i % details['eval_interval'] == 0:
        if i % details['max_steps'] == 0:
            # as done in call_main: checkpoint the converged models at each eval step.
            # flow_transfer bundles the encoder/decoder (autoencoder), flow model (velocity)
            # and regressor (safety_regressor); agent is the CBFTransfer model (reward critic/
            # value/actor trained on source, safety critic valid on both source and target).
            flow_transfer.save(modeldir, save_time)
            agent.save(modeldir, save_time)
            save_time += 1

    # Step 4: eval on Mt. No projection at deployment - Q_h^phi is already valid on target latents.
    # eval_info = evaluate_pr(agent, env_target, details['eval_episodes'])
    # wandb.log({f"{k}": v for k, v in eval_info.items()})
        if i % details['eval_interval'] == 0:
            eval_info = evaluate_pr(agent, env_target, details['eval_episodes'])
            print("eval_info at step", i, ":", eval_info)
            wandb.log({f"{k}": v for k, v in eval_info.items()}, step=i)


def main(_):
    parameters = FLAGS.config
    env_id = FLAGS.env_id
    mode = FLAGS.mode
    algo = 'fisor' if mode == 1 else 'tanh'
    # algo = 'tanh'
    parameters['project'] = FLAGS.project
    parameters['env_name'] = env_list[env_id]    
    parameters['group'] = parameters['env_name']
    # parameters['experiment_name'] = str(env_id) + '_'  + str(parameters['env_name']) + '_' + str(parameters['dataset_kwargs']['seed']) #str(np.random.randint(1000))
    parameters['experiment_name'] = str(env_id) + '_' + algo + ('_transfer' if FLAGS.transfer else '') + '_' + str(parameters['env_name']) + '_' + str(parameters['seed']) #str(np.random.randint(1000))

    if parameters['env_name'] == 'PointRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        # parameters['eval_episodes'] = 2
        # parameters['agent_kwargs']['cost_temperature'] = 2
        parameters['agent_kwargs']['reward_temperature'] = 5
        # parameters['agent_kwargs']['cost_tau'] = 0.01
        parameters['agent_kwargs']['cost_ub'] = 150
        parameters['agent_kwargs']['N'] = 8
    
    if env_id >= 21:  # Bullet safety gym envs
        parameters['agent_kwargs']['cost_limit'] = 5

    if FLAGS.transfer:
        # so config.json records which class the checkpoint actually is - viz_map.py and any
        # other loader dispatches on this field to know whether to rebuild a CBF or a CBFTransfer skeleton.
        parameters['agent_kwargs']['model_cls'] = 'CBFTransfer'

    # print(parameters)

    if not os.path.exists(f"./results/{parameters['env_name']}/{parameters['seed']}"):
        os.makedirs(f"./results/{parameters['env_name']}/{parameters['seed']}")
    with open(f"./results/{parameters['env_name']}/{parameters['seed']}/config.json", "w") as f:
        json.dump(to_dict(parameters), f, indent=4)
    
    # call_main(parameters, parameters['dataset_kwargs']['env_id'])
    if FLAGS.transfer:
        print('Running transfer pipeline...')
        call_main_transfer(parameters, env_id)
    else:
        call_main(parameters, env_id)
    # call_main(parameters)

if __name__ == '__main__':
    app.run(main)
