from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='130826_PR',
        # env_id=38,
        seed=-1,
        max_steps=500_001,
        eval_episodes=20,
        batch_size=512, #Actor batch size x 2 (so really 1024), critic is fixed to 256
        log_interval=1_000,
        eval_interval=250_000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=1.0,
        # env_id =38,
        # seed=0,
        pr_data='data/point_robot-expert-random-100k.hdf5', # The location of point_robot data (source, Ds)
        pr_data_target='data/pr_p4-50k.hdf5', # The location of the sparse target data (Dt); override with the real target-domain file
        # Hazard/goal geometry of the target CMDP. None reuses the source PointRobot's geometry
        # (a same-domain smoke test); set these once real target hazards are available.
        # Mirror-symmetric pair (point, -point), like the source's [0.4,-1.2]/[-0.4,1.2], but
        # relocated so the domain differs from source while still clearing the fixed start
        # (-1.8, 0.0) and goal (2.2, 2.2) by more than hazard_size.
        target_hazard_position_list=[[1.3, -1.0], [-1.3, 1.0]],
        target_hazard_size=0.8,
    )

    base_transfer_config = dict(
        transfer_kwargs=dict(
            latent_dim=32,
            hidden_dims=(256, 256),
            autoencoder_lr=1e-3,
            velocity_lr=1e-3,
            regressor_lr=1e-3,
            lambda_c=1.0,  # weight of the safety gap in the OT cost c(z0, z1)
            lambda_h=1.0,  # weight of L_align relative to L_FM
            ot_reg=0.05,   # Sinkhorn entropic regularization
            ot_iters=50,
            flow_steps=10,  # Euler steps used to integrate f_eta(z0) = psi_eta(z0, 1)
        ),
        transfer_schedule=dict(
            ot_batch_size=256,
            recon_steps=20_000,
            reg_steps=20_000,
            flow_train_steps=50_000,
        ),
    )


    '''
    17 - 0.75, [0.41 to 0.39]

    '''
    possible_structures = {
        "r": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    mode=1,  # FISOR
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cbf_lr=3e-4,
                    reward_temperature=3.0,
                    ddpm_temperature=0.3,
                    beta_schedule='linear',
                    T=3,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    value_layer_norm=False,
                    actor_tau=0.001,
                    reward_tau=0.9,
                    cost_tau=0.1,
                    cost_ub=150,
                    r_min=-0.001,
                    discount=0.99,         
                    # tanh_scale = 1.0,           
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_transfer_config,
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]

'''
not working well:
38 - {0.9,0.5,0.1,0.01}, {0.5,0.4,0.1,0.01}
'''
