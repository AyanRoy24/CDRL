from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='271225_BG',
        # env_id=38,
        seed=-1,
        max_steps=500_001,
        eval_episodes=20,
        batch_size=512, #Actor batch size x 2 (so really 1024), critic is fixed to 256
        log_interval=1_000,
        eval_interval=100_000,
        normalize_returns=True,
        # Soft-constraint budgets used at evaluation (budget-conditioned states).
        eval_cost_thresholds=(20, 40, 80),
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=1.0,
        # env_id =38,
        # seed=0,
        pr_data='jaxrl5/data/point_robot-expert-random-100k.hdf5', 
        # tgt_data='/home2/ayanroy/data/copdt/SafetyAntVelocityGymnasium-v0-250-2249.hdf5', 
        # tgt_data='/home2/ayanroy/data/copdt/SafetyHalfCheetahVelocityGymnasium-v0-250-2495.hdf5', 
        # tgt_data='/home2/ayanroy/data/copdt/SafetyHopperVelocityGymnasium-v0-250-2240.hdf5', 
        # tgt_data='/home2/ayanroy/data/copdt/SafetySwimmerVelocityGymnasium-v0-200-1686.hdf5', 
        # tgt_data='/home2/ayanroy/data/copdt/SafetyWalker2dVelocityGymnasium-v0-300-2729.hdf5', 
        tgt_data=None, 
    )


    '''
    17 - 0.75, [0.41 to 0.39]

    '''
    possible_structures = {
        "r": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                   
                    # cost_temperature=5,
                    reward_temperature=3,
                   
                    # T=5,
                    N=16,
                   
                    # M=0,
                    # clip_sampler=True,
                    # actor_dropout_rate=0.1,
                    # actor_num_blocks=3,
                   
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    
                    # actor_layer_norm=True,
                    value_layer_norm=False,
                    actor_tau=0.001,
                   
                    # actor_architecture='ln_resnet',
                    # critic_objective='expectile',
                    reward_tau  = 0.75,
                    cost_tau = 0.15,
                   
                    # critic_type="hj", #[hj, qc]
                    cost_ub=150,
                   
                    # beta_schedule='vp',
                    # actor_objective="feasibility", 
                    # sampling_method="ddpm", 
                    # extract_method="minqc", 
                    discount=0.99,

                    critic_layer_norm=False,
                    cbf_lr=3e-4,
                    r_min=-0.001,
                    # tanh_scale = 5.0,
                    mode=1,  # FISOR
                    # num_components=9,

                    # ---- budget-conditioned soft constraints (cf. CBF/bcrl) ----
                    cost_temperature=5,      # AWR temperature for the safe fallback actor
                    cost_discount=0.99,      # discount for the cumulative cost critic Q_C
                    cost_expectile=0.3,      # in-sample (optimistic) expectile for V_C
                    budget_max=100.0,        # max budget the networks are conditioned on
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]

'''
not working well:
38 - {0.9,0.5,0.1,0.01}, {0.5,0.4,0.1,0.01}
'''
