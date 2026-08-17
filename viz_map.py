import os
import sys
# sys.path.append('.')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from absl import app, flags
import re
import json
import numpy as np
from ml_collections import config_flags, ConfigDict
import matplotlib.pyplot as plt
from matplotlib import colors
import jax
from point_robot import PointRobot
from cbf import CBF
from cbf_transfer import CBFTransfer
from flow_transfer import FlowTransfer

FLAGS = flags.FLAGS
flags.DEFINE_string('model_location', 'results/PointRobot/372/', 'model location for point robot model')

def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d

# hazard_position_list = [np.array([0.4, -1.2]), np.array([-0.4, 1.2])]
# hazard_position_list = [np.array([0.3, -1.0]), np.array([-0.7, 1.3])] #1
# hazard_position_list = [np.array([0.1, -1.2]), np.array([-0.7, 1.2])] #2
# hazard_position_list = [np.array([1.1, -0.2]), np.array([-1.7, 0.2])] #3
hazard_position_list = [np.array([1.3, -1.0]), np.array([-1.3, 1.0])] #4

label_size = 18
legend_size = 30
ticks_size = 18
location = -0.3
width = 0.5

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': label_size}

def plot_pr_pic(ax, agent, v, theta, cb=False, hazard_position_list=hazard_position_list):
    ## generate batch obses
    x1 = np.linspace(-3.0, 3.0, 201)
    x2 = np.linspace(-3.0, 3.0, 201)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    batch_obses = np.zeros((len(flatten_x1), 11), dtype=np.float32)  # (201*201, 11)
    assert batch_obses.shape == (201*201, 11)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2

    batch_obses[:, 2] = v * np.ones_like(flatten_x1)
    thetas = theta * np.ones_like(flatten_x1)
    batch_obses[:, 3] = np.cos(thetas)
    batch_obses[:, 4] = np.sin(thetas)

    c = np.cos(theta)
    s = np.sin(theta)

    rot_mat = np.array([[c, -s],
                        [s, c]], dtype=np.float32)

    k = 0

    for hazard_pos in hazard_position_list:

        pos = (hazard_pos[:2] - batch_obses[:,:2]) @ rot_mat  # (B, 2)
        x = pos[:,0]
        y = pos[:,1]
        hazard_vec = x + 1j * y

        dist = np.abs(hazard_vec)
        angle = np.angle(hazard_vec)

        batch_obses[:,5+k*3] = dist
        batch_obses[:,6+k*3] = np.cos(angle)
        batch_obses[:,7+k*3] = np.sin(angle)

        k += 1
    
    '''
    safe value
    '''
    safe_value = agent.safe_value.apply_fn({"params": agent.safe_value.params}, jax.device_put(batch_obses))
    value = safe_value

    value_flatten = np.asarray(value)
    value_square = value_flatten.reshape(x1_grid.shape)
        
    '''
    draw hj
    '''
    norm = colors.Normalize(vmin=-3.5, vmax=1.01)
    
    ct = ax.contourf(
        x1_grid, x2_grid, value_square,
        norm=norm,
        levels=30,
        cmap='rainbow',
    )

    ct_line = ax.contour(
        x1_grid, x2_grid, value_square,
        levels=[0], colors='#32ABD6',
        linewidths=2.0, linestyles='solid'
    )
    ax.clabel(ct_line, inline=True, fontsize=15, fmt=r'0',)

    if cb==True:
        cb = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.02, ticks=np.linspace(-3.2, 0.8, 6))
        cb.ax.tick_params(labelsize=ticks_size)

        cbarlabels = cb.ax.get_yticklabels() 
        [label.set_fontname('Times New Roman') for label in cbarlabels]

    arrow_x1 = np.linspace(-1.8, 1.8, 3)
    arrow_x2 = np.linspace(-1.8, 1.8, 3)
    ax1_grid, ax2_grid = np.meshgrid(arrow_x1, arrow_x2)

    thetas = theta * np.ones_like(ax1_grid)
    ux = v * np.cos(thetas)
    uy = v * np.sin(thetas)
    ax.quiver(arrow_x1,arrow_x2,ux,uy,color='k',angles='xy', scale_units='xy', scale=2,alpha=0.5)
    return ax


def plot_pic(env, agent, model_location):

    fig, ([ax1,ax2,ax3,ax4]) = plt.subplots(
        nrows=1, ncols=4,
        figsize=(10.5, 2.5),
        constrained_layout=True,
    )
    
    my_x_ticks = np.arange(-3,3.01,1.5)
    my_y_ticks = np.arange(-3,3.01,1.5)

    labels = ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels() \
    + ax3.get_xticklabels() + ax3.get_yticklabels() + ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    '''
    subplot1 : plot the task
    '''
    ax1 = env.plot_task(ax1)
    ax1.set_xticks(my_x_ticks)
    ax1.set_yticks(my_y_ticks)
    ax1.set_xlim((-3, 3))  
    ax1.set_ylim((-3, 3))  
    ax1.tick_params(labelsize=ticks_size)
    ax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1.spines['top'].set_color('gray')      
    ax1.spines['bottom'].set_color('gray')  
    ax1.spines['left'].set_color('gray')   
    ax1.spines['right'].set_color('gray')  

    
    '''
    subplot2,3,4 : plot the feasible region and the learned feasible region for different v and theta
    '''
    # Query the barrier relative to env's own hazards, so a target-domain env (transfer runs)
    # automatically featurizes states the same way the target-valid Q_h^phi expects.
    ax2 = plot_pr_pic(ax2, agent, v=0.5, theta=np.pi / 4, hazard_position_list=env.hazard_position_list)
    ax2 = env.plot_map(ax2, v=0.5, theta=np.pi / 4)

    ax3 = plot_pr_pic(ax3, agent, v=1, theta=np.pi / 2, hazard_position_list=env.hazard_position_list)
    ax3 = env.plot_map(ax3, v=1, theta=np.pi / 2)

    ax4 = plot_pr_pic(ax4, agent, v=1.5, theta=np.pi / 4, cb=True, hazard_position_list=env.hazard_position_list)
    ax4 = env.plot_map(ax4, v=1.5, theta=np.pi / 4)

    
    for ax in [ax2, ax3, ax4]:
        ax.set_xticks(my_x_ticks)
        ax.set_yticks(my_y_ticks)
        ax.set_xlim((-3, 3))
        ax.set_ylim((-3, 3))
        ax.tick_params(labelsize=ticks_size)
        ax.set_xlim([-2.7,2.7])
        ax.set_ylim([-2.7,2.7])
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.spines['bottom'].set_linewidth(width)
        ax.spines['left'].set_linewidth(width)
        ax.spines['right'].set_linewidth(width)
        ax.spines['top'].set_linewidth(width)
        ax.spines['top'].set_color('white') 
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white') 
        ax.spines['right'].set_color('white')
    
    plt.savefig(f"{model_location}/imgs/viz_map.png", dpi=600)


def load_diffusion_model(model_location):

    with open(os.path.join(model_location, 'config.json'), 'r') as file:
        cfg = to_config_dict(json.load(file))

    # `transfer_kwargs` is only ever written for --transfer runs, so it's a reliable signal
    # even for older checkpoints saved back when config.json still hardcoded model_cls="CBF".
    is_transfer = 'transfer_kwargs' in cfg

    if is_transfer:
        data_kwargs = cfg['dataset_kwargs']
        target_hazards = data_kwargs.get('target_hazard_position_list')
        if target_hazards is not None:
            target_hazards = [np.array(p) for p in target_hazards]
        env = PointRobot(id=0, seed=0, hazard_position_list=target_hazards, hazard_size=data_kwargs.get('target_hazard_size'))
    else:
        env = eval('PointRobot')(id=0, seed=0)

    config_dict = dict(cfg['agent_kwargs'])
    model_cls = config_dict.pop("model_cls")

    if is_transfer:
        transfer_kwargs = dict(cfg['transfer_kwargs'])
        # Only used to get correctly-shaped autoencoder/velocity TrainStates to hang the
        # checkpoint's real weights on - agent.load() below overwrites every leaf anyway.
        flow_skeleton = FlowTransfer.create(cfg['seed'], env.observation_space, env.action_space, **transfer_kwargs)
        agent = CBFTransfer.create(
            cfg['seed'], env.observation_space, env.action_space,
            autoencoder=flow_skeleton.autoencoder,
            velocity=flow_skeleton.velocity,
            latent_dim=transfer_kwargs['latent_dim'],
            flow_steps=transfer_kwargs['flow_steps'],
            **config_dict,
        )
    else:
        agent = globals()[model_cls].create(
            cfg['seed'], env.observation_space, env.action_space, **config_dict
        )

    def get_model_file():
        # Prefix-filtered: a transfer run's directory also holds flow_transferN.pickle files
        # (a FlowTransfer, not an agent), which share the same numeric suffix space as modelN.pickle
        # and would otherwise be picked at random depending on os.listdir's return order.
        files = os.listdir(f"{model_location}")
        pickle_files = [f for f in files if f.startswith('model') and f.endswith('.pickle')]
        numbers = {}
        for file in pickle_files:
            match = re.search(r'\d+', file)
            number = int(match.group())
            path = os.path.join(f"{model_location}", file)
            numbers[number] = path

        max_number = max(numbers.keys())
        max_path = numbers[max_number]
        return max_path

    model_file = get_model_file()
    new_agent = agent.load(model_file)

    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")

    return env, new_agent

def main(_):

    env, diffusion_agent = load_diffusion_model(FLAGS.model_location)
    
    plot_pic(env, diffusion_agent, FLAGS.model_location)


if __name__ == '__main__':
    print('current working directory:', os.getcwd())
    app.run(main)
