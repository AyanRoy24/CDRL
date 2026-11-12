"""Config of the DSRL / CoPDT pipeline (train_sa_encoder.py -> train_context_encoder.py ->
train_general.py).  The previous COSTA multitask-ant config is train_config_costa.py.

Stage 0  sa_encoder_kwargs : per-family state / action auto-encoders        (CoPDT sa_encoder_configs.py)
Stage 1  context_kwargs    : CoPDT context encoder, no discriminator        (CoPDT context_encoder_configs.py)
Stage 2  agent_kwargs      : budget-adaptive IQL (iql_pooled.IQLPooled), trained on the pooled
                             latent data of every task, conditioned on zeta and on the budget
"""
from ml_collections import ConfigDict
import numpy as np


def get_config(config_string):
    base_real_config = dict(
        project='CoPDT-DSRL',
        seed=-1,
        max_steps=500_001,
        eval_episodes=5,                 # per task and budget; 26 tasks x 4 budgets x 1000 steps per eval
        batch_size=512,                  # transitions per update = 2 * batch_size, spread over the tasks
        log_interval=1_000,
        eval_interval=100_000,
        # Soft-constraint budgets rolled out at evaluation: CoPDT's multi-budget protocol.
        eval_cost_thresholds=(10, 20, 40, 80),
    )
    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    dsrl_kwargs = dict(
        cost_limit=10,                   # target cost the DSRL envs are created with (CoPDT's default)
        reward_norm=1.0,                 # per-step reward of an expert, after the per-task rescaling
    )

    # ---- Stage 0: shared latent space (train_sa_encoder_share.py defaults) ----------------------
    sa_encoder_kwargs = dict(
        state_encode_dim=32,
        action_encode_dim=2,             # families whose action dim is already 2 skip the action AE
        state_encoder_hidden_sizes=(128, 128, 128),
        inverse_dynamics_model_hidden_sizes=(128, 128, 128),
        action_encoder_hidden_sizes=(128, 128, 128),
        idm_loss_weight=0.1,
        add_noise=True,
        noise_scale=0.1,
        learning_rate=1e-4,
        batch_size=2048,
        state_epoch_num=100,
        action_epoch_num=100,
        steps_per_epoch=1000,
    )

    # ---- Stage 1: CoPDT context encoder (train_context_encoder.py defaults) ----------------------
    context_kwargs = dict(
        context_encoding_dim=16,
        context_encoder_hidden_sizes=(256, 256, 256),
        feature_hidden_sizes=(256, 128),
        decoder_hidden_sizes=(128, 128),
        simple_gate=False,
        simple_mlp=False,
        state_loss_weight=1.0,
        reward_loss_weight=0.0,
        cost_loss_weight=2.0,
        context_size=500,                # transitions per context set
        meta_batch_per_task=4,           # context sets per task per update
        reward_scale=0.1,                # scale of the reward fed to the context encoder
        learning_rate=1e-4,
        min_learning_rate=3e-5,
        betas=(0.9, 0.999),
        decay_step=2000,
        decay_rate=0.9,
        update_steps=100_000,
        eval_every=1000,
    )

    possible_structures = {
        "r": ConfigDict(
            dict(
                agent_kwargs=dict(
                    # budget-adaptive IQL (CBF/bcrl bcrl_det.IQL) hyperparameters
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cbf_lr=3e-4,             # Q_C / V_C learning rate
                    reward_temperature=8,    # AWR temperature beta
                    reward_tau=0.8,          # reward expectile
                    cost_expectile=0.3,      # in-sample (optimistic) expectile for V_C
                    discount=0.99,
                    cost_discount=0.99,
                    actor_weight_decay=None,
                    decay_steps=None,        # None: cosine-decay the actor lr over max_steps
                    value_layer_norm=False,
                    delta_max=100.0,         # upper end of the per-minibatch budget sample
                ),
                dsrl_kwargs=dsrl_kwargs,
                sa_encoder_kwargs=sa_encoder_kwargs,
                context_kwargs=context_kwargs,
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]
