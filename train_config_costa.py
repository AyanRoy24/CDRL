from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='130826_PR',
        # env_id=38,
        seed=-1,
        max_steps=500_001,
        eval_episodes=10,
        batch_size=512, #Actor batch size x 2 (so really 1024), critic is fixed to 256
        log_interval=1_000,
        eval_interval=100_000,
        normalize_returns=True,
        # Soft-constraint budgets used at evaluation (budget-conditioned states).
        eval_cost_thresholds=(10, 20, 40, 80),
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=1.0,
        # env_id =38,
        # seed=0,
        src_data='/home2/ayanroy/data/costa/', 
        # tgt_data='/home2/ayanroy/data/costa/cheetah-vel-safe_/', # The location of the sparse target data (Dt); override with the real target-domain file
        # target_hazard_position_list=[[1.3, -1.0], [-1.3, 1.0]],  # 4
        # target_hazard_position_list=[[1.1, -0.2], [-1.7, 0.2]],  # 3
        # target_hazard_size=0.8,
    )

    base_transfer_config = dict(
        transfer_kwargs=dict(
            latent_dim=32,
            hidden_dims=(256, 256),
            autoencoder_lr=1e-3,
            velocity_lr=1e-3,
            regressor_lr=1e-3,
            lambda_c=9.82,# 9.81564626268391,  # weight of the safety gap in the OT cost c(z0, z1)
            lambda_h=0.22, #0.2205050207394007,  # weight of L_align relative to L_FM
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
        # Stage 3: weight of the real-target-data term (lambda_t) in the safety critic loss
        # phi <- argmin_phi E_Ms[|Qh(zs)-yhs| + |Qh(f_eta(zs))-yhs|] + lambda_t * E_Mt[|Qh(zt)-yht|]
        lambda_t=0.01 # 0.010920807120331886,
    )

    # -------------------------------------------------------------------------
    # Part III (Sec. `sec:pooled`): pooled multi-source training with an inferred
    # constraint context.  Triggered when --src_goal lists more than one goal id
    # for a multitask-ant transfer run: each goal is one constraint regime c^(k),
    # loaded from its <env_name>_<goal_id>.hdf5 buffer.  Stage step counts
    # (recon/reg/flow) are taken from `transfer_schedule` above.
    # See models/data/pooled_datasets.py and models/agents/cbf/cbf_pooled_g.py.
    # -------------------------------------------------------------------------
    base_pooled_config = dict(
        pooled_kwargs=dict(
            # --- Stage 1: context encoder q_xi -----------------------------
            # Matches CDSRL/COSTA/COSTA's default context encoder exactly
            # (`encoder_type="mlp_attn"`, `nets/encoder.py::MLPAttnEncoder` +
            # `nets/decoder.py::Decoder`, trained via `OfflineMetaLearner`'s
            # distance-metric + decoder-reconstruction objective, not SupCon).
            seg_len=150,                     # segment length m for q_xi
            context_dim=8,                  # keep small (Sec. `sec:pooled-practical`, Stage 1)
            ctx_hidden_dims=(256, 256),
            ctx_lr=1e-3,
            ctx_steps=20_000,
            ctx_batch_per_regime=64,        # segments per regime per contrastive minibatch
            cost_anchor_frac=0.5,           # fraction of segments containing a non-zero cost
            decoder_hidden_dims=(256, 256), # COSTA's `Decoder` (predicts the per-step cost bit)
            decoder_weight=0.5,             # COSTA's `decoder_weight`
            # COSTA's cost-dynamics term (run_context_encoding.py passes the per-task cost
            # discriminators): the encoder is also trained on segments whose cost bits were relabeled by
            # another regime's discriminator, so it must read the cost column instead of the state
            # statistics.  Needs <costa_dir>/run/<env>_<goal>/discriminator/best_model.pt for every
            # --src_goal (COSTA has them for ant-dir-safe, ant-circle-safe, half-cheetah-vel-safe,
            # goals 0-2); set ctx_relabel=False (--config.pooled_kwargs.ctx_relabel=False) otherwise.
            ctx_relabel=True,
            costa_dir='/home2/ayanroy/COSTA/COSTA',
            # weight of the counterfactual term.  COSTA uses 0.1 with 300-transition contexts; on this
            # pipeline's short windows 0.1 leaves the cost column ignored (seg_len=16 even at 1.0 only
            # partly follows it, seg_len~150 at 1.0 follows it ~90%+), so it is 1.0 here.
            ctx_contrastive_weight=1.0,
            relabel_device='cpu',           # where the discriminators label the datasets (cached in ./relabel_cache)
            # --- Stage 3/5: bcrl_det.IQL (replaces the budget-adaptive-CBF
            # additive Q_C/Q_h bookkeeping of CBFPooled) --------------------
            delta_max=100.0,                 # upper end of the per-minibatch budget sample
            delta_init=10.0,                # eval-time budget under test (also see eval_cost_thresholds)
        ),
    )

    # -------------------------------------------------------------------------
    # adapt_flow.py: target adaptation of a saved general policy.  The autoencoder/
    # flow hyperparameters and step counts come from transfer_kwargs/transfer_schedule.
    # -------------------------------------------------------------------------
    base_adapt_config = dict(
        adapt_kwargs=dict(
            pool_per_regime=100_000,  # latents precomputed per source regime / for the target
            ode_method='midpoint',    # any torchdiffeq method; flow_steps is its grid unless adaptive
            critic_steps=50_000,
            critic_batch_size=256,
            critic_lr=3e-4,
            lambda_t=1.0,             # weight of the TD term on real target data
            eval_interval=25_000,
        ),
    )

# Best params: {'lambda_c': 9.81564626268391, 'lambda_h': 0.2205050207394007, 'lambda_t': 0.010920807120331886}
# Best trial eval_info: {'return': 2.48906354970297, 'episode_len': 36.0, 'cost': 9.0, 'coverage': 0.8055555820465088, 'validity': 0.9722222685813904}

    possible_structures = {
        "r": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    mode=1,  # FISOR
                    cost_limit=25,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cbf_lr=3e-4,
                    reward_temperature=8,
                    ddpm_temperature=0.3,
                    beta_schedule='linear',
                    T=3,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_weight_decay=None,
                    decay_steps=int(3e6), # match max_steps for bcrl, this is fisor value
                    value_layer_norm=False,
                    actor_tau=0.001,
                    reward_tau=0.8,
                    cost_tau=0.1,
                    cost_ub=150,
                    r_min=-0.001,
                    discount=0.99,
                    # tanh_scale = 1.0,

                    # ---- budget-conditioned soft constraints (cf. CBF/direct) ----
                    cost_temperature=5,      # AWR temperature for the safe fallback actor
                    cost_discount=0.99,      # discount for the cumulative cost critic Q_C
                    cost_expectile=0.3,      # in-sample (optimistic) expectile for V_C
                    budget_max=100.0,        # max budget the networks are conditioned on
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_transfer_config,
                **base_pooled_config,
                **base_adapt_config,
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]

