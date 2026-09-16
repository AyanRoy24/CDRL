import matplotlib.pyplot as plt

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
from env.env_list import env_list
from env.point_robot import PointRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import CBF 
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_md, evaluate_pr #, offline_evaluation , plot_cbf_cost_vs_safe_value, calculate_coverage
import json

FLAGS = flags.FLAGS
# flags.DEFINE_string("config", "train_config.py:r", "Config file")
flags.DEFINE_integer('env_id', 23, 'Choose env')
# flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_integer('mode', 1, 'Mode for training')
flags.DEFINE_integer('max_steps', 500_001, 'max steps')
# flags.DEFINE_integer('seed', -1, 'random seed')
flags.DEFINE_float('cost_tau', 0.15, 'cost tau')
flags.DEFINE_float('reward_tau', 0.75, 'reward tau')
# flags.DEFINE_integer('eval', 10000, 'eval steps')
flags.DEFINE_string('project', '081125', 'Name of the experiment')
flags.DEFINE_string('eval_budgets', '', 'Comma-separated soft-constraint budgets for eval, e.g. "20,40,80"')

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
        ds = DSRLDataset(env, cost_scale=details['dataset_kwargs']['cost_scale'], data_location=details['dataset_kwargs']['tgt_data'] if details['dataset_kwargs']['tgt_data'] is not None else None)#, ratio=details['ratio'])
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
    if details['agent_kwargs']['mode'] == 4: # reward leakage plot
        agent = agent.load('results/OfflineSwimmerVelocityGymnasium-v1/491/model20.pickle') 
        # CBF/SCBF_og/results/OfflineCarButton1Gymnasium-v0/337/model1.pickle #MSE - 158, TD - 337, huber- 565
        # CBF/SCBF_og/results/OfflineSwimmerVelocityGymnasium-v1/491/model20.pickle
    # Sample 10,000 state-action pairs from the offline dataset
        # ds = DSRLDataset(gym.make("OfflineSwimmerVelocityGymnasium-v1"))
        batch = ds.sample_jax(10_000)
        observations = batch["observations"]
        actions = batch["actions"]

        # Compute Q_h and Q_r
        Q_h = agent.safe_critic.apply_fn({'params': agent.safe_critic.params}, observations, actions).max(axis=0)
        Q_r = agent.critic.apply_fn({'params': agent.critic.params}, observations, actions).min(axis=0)

        # Plot
        plt.figure(figsize=(1.65, 1.8))
        plt.rcParams.update({
            'font.size': 8.5,
            'axes.labelsize': 9.5,
            'xtick.labelsize': 7.5,      # Safely above the 7pt absolute floor
            'ytick.labelsize': 7.5,
            'legend.fontsize': 7.0,      # Minimum limit allowed by IJCAI
            'text.usetex': False,       # <--- TURN THIS OFF
            'mathtext.fontset': 'stix',  # Use STIX fonts for math
        })
        plt.scatter(Q_h, Q_r, alpha=0.5, s=4)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.0, label='$Q_h=0$')
        plt.title(r'$\mathrm{CBF}_{\mathrm{smooth}}$', fontsize=9.5)
        plt.xlabel(r'$Q_h(s,a)$')
        plt.ylabel(r'$Q_r(s,a)$')
        # plt.title('Reward Critic vs Safety Critic')
        # plt.legend()
        plt.tight_layout(pad=0.1)
        plt.savefig('RL_plot_left.png', format='png', dpi=400, bbox_inches='tight')        
        # plt.show()
    else:
        save_time, eval_num = 1, 1
        # agent.save(f"./results/{details['group']}/{details['experiment_name']}", save_time)                
        for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
            sample = ds.sample_jax(details['batch_size'])     
            agent, info = agent.update(sample)
            if i % details['log_interval'] == 0:
                wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)
            # if i >= (details['max_steps']-20):
            if i== details['max_steps'] -1 and details['env_name'] == 'PointRobot':
                agent.save(f"./results/{details['env_name']}/{details['seed']}", save_time)                
                            
            if i % details['eval_interval'] == 0:
                # save_time += 1
                if details['env_name'] == 'PointRobot':
                    eval_info = evaluate_pr(agent, env, details['eval_episodes'])
                # elif env_id >= 30:
                #     eval_info = evaluate_md(obs_mean, obs_std, details['seed'], env_id, eval_num, agent, env, details['eval_episodes'], render=False) #, save_video=True, )
                #         # eval_num += 1
                else:
                    eval_info = evaluate(details['seed'], agent, env, details['eval_episodes'],
                                         cost_thresholds=tuple(details['eval_cost_thresholds']))

                # if details['env_name'] != 'PointRobot' :#and 'n_return' not in eval_info:
                    eval_info["n_return"], eval_info["n_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])

                print ({f"eval/n_ret": eval_info["n_return"], f"eval/n_cost": eval_info["n_cost"]})
                wandb.log({f"{k}": v for k, v in eval_info.items()} , step=i)        
 

def main(_):
    parameters = FLAGS.config
    env_id = FLAGS.env_id
    mode = FLAGS.mode
    parameters['max_steps'] = FLAGS.max_steps
    # parameters['seed'] = FLAGS.seed
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
    parameters['env_name'] = env_list[FLAGS.env_id]
    parameters['group'] = parameters['env_name']
    parameters['env_name'] = env_list[env_id]    
    # parameters['experiment_name'] = str(env_id) + '_'  + str(parameters['env_name']) + '_' + str(parameters['dataset_kwargs']['seed']) #str(np.random.randint(1000))
    parameters['experiment_name'] = str(env_id) + '_' + algo + '_' + str(parameters['env_name']) + '_' + str(parameters['seed']) #str(np.random.randint(1000))
    
    if env_id >= 21:  # Bullet safety gym envs
        parameters['agent_kwargs']['cost_limit'] = 5

    if parameters['env_name'] == 'PointRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        parameters['agent_kwargs']['cost_temperature'] = 2
        parameters['agent_kwargs']['reward_temperature'] = 5
        parameters['agent_kwargs']['cost_ub'] = 150
        parameters['agent_kwargs']['N'] = 8
    # print(parameters)

    # if not os.path.exists(f"./results/{parameters['env_name']}/{parameters['seed']}"):
    #     os.makedirs(f"./results/{parameters['env_name']}/{parameters['seed']}")
    # with open(f"./results/{parameters['env_name']}/{parameters['seed']}/config.json", "w") as f:
    #     json.dump(to_dict(parameters), f, indent=4)
    
    # call_main(parameters, parameters['dataset_kwargs']['env_id'])
    call_main(parameters, env_id)

def reward_leakage_plot(_):
    # Load model
    with open('results/OfflineSwimmerVelocityGymnasium-v1/491/model20.pickle', 'rb') as f:
        state_dict = pickle.load(f)
    agent = CBF.load('results/OfflineSwimmerVelocityGymnasium-v1/491/model20.pickle')

     # Sample 10,000 state-action pairs from the offline dataset
    ds = DSRLDataset(gym.make("OfflineSwimmerVelocityGymnasium-v1"))
    batch = ds.sample_jax(10_000)
    observations = batch["observations"]
    actions = batch["actions"]

    # Compute Q_h and Q_r
    Q_h = agent.safe_critic.apply_fn({'params': agent.safe_critic.params}, observations, actions).max(axis=0)
    Q_r = agent.critic.apply_fn({'params': agent.critic.params}, observations, actions).min(axis=0)

    # Plot
    plt.scatter(Q_h, Q_r, alpha=0.5)
    plt.axvline(0, color='red', linestyle='--', label='Safety boundary (Q_h=0)')
    plt.xlabel('Q_h(s,a) (Safety Critic)')
    plt.ylabel('Q_r(s,a) (Reward Critic)')
    plt.title('Reward Critic vs Safety Critic')
    plt.legend()
    plt.savefig("reward_leakage_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    app.run(main)
    # app.run(reward_leakage_plot)
