"""Stage 3 (+5) of Sec. `sec:pooled`, replacing the budget-adaptive-CBF training
of `CBFPooled` (`cbf_pooled_g.py`) with CBF/bcrl's IQL algorithm
(`CBF/bcrl/bcrl_det.py`) directly: a budget-conditioned cost/reward critic pair
and a dual (budget-conditioned + safe-fallback) actor, trained with in-sample
expectile regression + advantage-weighted regression, exactly as `bcrl_det.IQL`.

There is no hard reachability barrier `Q_h` here and no latent autoencoder/OT
flow: feasibility is handled purely by bcrl_det's own budget/safe-actor switch
(`get_action`'s `cost_limits > cost_lb` gate), operating directly on raw
observations. On top of vanilla bcrl_det, every network is additionally
conditioned on the pooled constraint context `zeta` inferred by the frozen
Stage-1 encoder `q_xi` (`ContextEncoderTrainer`) -- `zeta` is concatenated to the
observation wherever bcrl_det would read `observations`, exactly like
`CBFPooled` context-conditions its own critics.

Setting `context_dim = 0` (an all-empty `zeta`) recovers plain single-regime
bcrl_det.
"""

import os
import pickle
from functools import partial
from typing import Optional, Sequence

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from models.agents.cbf.cbf_g import expectile_loss
from models.networks import Ensemble, GaussianPolicy, MLP, StateActionValue, StateValue, get_weight_decay_mask


def _cat(*xs):
    return jnp.concatenate(xs, axis=-1)


def _sample_deltabar(key, shape, base_cost, max_cost):
    """delta ~ U[base_cost, max_cost], base_cost clipped into [0, max_cost]
    (`bcrl_det.IQL.sample_deltabar`)."""
    base_cost = jnp.clip(base_cost, 0.0, max_cost)
    u = jax.random.uniform(key, shape)
    span = jnp.clip(max_cost - base_cost, 0.0)
    return u * span + base_cost


class DualBudgetPolicy(nn.Module):
    """`bcrl_det.GaussianHindSightPolicy`: a budget-conditioned policy and an
    independent zero-budget safe-fallback policy, as two separate parameter
    sets under one Module (so both are updated by one `actor.apply_gradients`)."""

    hidden_dims: Sequence[int]
    action_dim: int
    tanh_squash_distribution: bool = False

    @nn.compact
    def __call__(self, observations, delta=None, temperature: float = 1.0):
        dist = GaussianPolicy(
            self.hidden_dims, self.action_dim, tanh_squash_distribution=self.tanh_squash_distribution
        )(observations, budget=delta, temperature=temperature)
        dist_safe = GaussianPolicy(
            self.hidden_dims, self.action_dim, tanh_squash_distribution=self.tanh_squash_distribution
        )(observations, budget=None, temperature=temperature)
        return dist, dist_safe


class IQLPooled(struct.PyTreeNode):
    rng: jnp.ndarray

    # --- frozen constraint-context encoder q_xi -------------------------------
    context_encoder: TrainState = None
    context_dim: int = struct.field(pytree_node=False, default=8)
    seg_m: int = struct.field(pytree_node=False, default=16)

    # --- bcrl_det.IQL networks, all context-conditioned -----------------------
    cost_critic: TrainState = None       # Q_C(s, zeta, a)
    cost_target_critic: TrainState = None
    cost_value: TrainState = None        # V_C(s, zeta)

    critic: TrainState = None            # Q_r(s, zeta, a, delta)
    target_critic: TrainState = None
    value: TrainState = None             # V_r(s, zeta, delta)

    actor: TrainState = None             # pi(a|s,zeta,delta) + safe fallback pi_safe(a|s,zeta)

    # --- bcrl_det.IQLConfig hyperparameters ------------------------------------
    discount: float = 0.99
    cost_discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.8
    cost_expectile: float = 0.3
    beta: float = 8.0
    delta_max: float = 50.0

    # --- deployment runtime state ----------------------------------------------
    ctx: jnp.ndarray = None
    delta_hat: float = 0.0

    # ---------------------------------------------------------------- factory
    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        context_encoder: TrainState,
        context_dim: int,
        seg_m: int = 16,
        hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        cost_critic_lr: float = 3e-4,
        cost_value_lr: float = 3e-4,
        discount: float = 0.99,
        cost_discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        cost_expectile: float = 0.3,
        beta: float = 8.0,
        delta_max: float = 50.0,
        actor_weight_decay: Optional[float] = None,
        decay_steps: Optional[int] = None,
        critic_layer_norm: bool = True,
        value_layer_norm: bool = False,
        tanh_squash_distribution: bool = False,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, cost_critic_key, cost_value_key = jax.random.split(rng, 6)

        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        z_dim = obs_dim + context_dim

        obs_ctx = jnp.zeros((1, z_dim))
        actions = jnp.zeros((1, action_dim))
        delta0 = jnp.zeros((1, 1))

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        # --- Q_C(s,zeta,a), V_C(s,zeta) ----------------------------------------
        cost_critic_base = partial(MLP, hidden_dims=hidden_dims, activate_final=True, use_layer_norm=critic_layer_norm)
        cost_critic_def = Ensemble(partial(StateActionValue, base_cls=cost_critic_base), num=2)
        cost_critic_params = FrozenDict(cost_critic_def.init(cost_critic_key, obs_ctx, actions)["params"])
        cost_critic = TrainState.create(apply_fn=cost_critic_def.apply, params=cost_critic_params, tx=optax.adam(cost_critic_lr))
        cost_target_critic = TrainState.create(
            apply_fn=cost_critic_def.apply, params=cost_critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        cost_value_base = partial(MLP, hidden_dims=hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        cost_value_def = StateValue(base_cls=cost_value_base)
        cost_value_params = FrozenDict(cost_value_def.init(cost_value_key, obs_ctx)["params"])
        cost_value = TrainState.create(apply_fn=cost_value_def.apply, params=cost_value_params, tx=optax.adam(cost_value_lr))

        # --- Q_r(s,zeta,a,delta), V_r(s,zeta,delta) -----------------------------
        critic_base = partial(MLP, hidden_dims=hidden_dims, activate_final=True, use_layer_norm=critic_layer_norm)
        critic_def = Ensemble(partial(StateActionValue, base_cls=critic_base), num=2)
        critic_params = FrozenDict(critic_def.init(critic_key, obs_ctx, actions, delta0)["params"])
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(critic_lr))
        target_critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        value_base = partial(MLP, hidden_dims=hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base)
        value_params = FrozenDict(value_def.init(value_key, obs_ctx, delta0)["params"])
        value = TrainState.create(apply_fn=value_def.apply, params=value_params, tx=optax.adam(value_lr))

        # --- pi(a|s,zeta,delta) + pi_safe(a|s,zeta) -----------------------------
        actor_def = DualBudgetPolicy(actor_hidden_dims, action_dim, tanh_squash_distribution=tanh_squash_distribution)
        actor_params = FrozenDict(actor_def.init(actor_key, obs_ctx, delta0)["params"])
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adamw(
                learning_rate=actor_lr,
                weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                mask=get_weight_decay_mask,
            ),
        )

        return cls(
            rng=rng,
            context_encoder=context_encoder,
            context_dim=context_dim,
            seg_m=seg_m,
            cost_critic=cost_critic,
            cost_target_critic=cost_target_critic,
            cost_value=cost_value,
            critic=critic,
            target_critic=target_critic,
            value=value,
            actor=actor,
            discount=discount,
            cost_discount=cost_discount,
            tau=tau,
            expectile=expectile,
            cost_expectile=cost_expectile,
            beta=beta,
            delta_max=delta_max,
            ctx=jnp.zeros((context_dim,)),
            delta_hat=0.0,
        )

    # ----------------------------------------------------------------- helpers
    def infer_context(self, segments):
        """zeta_hat = q_xi(sigma). Frozen: no gradient ever flows into q_xi."""
        z = self.context_encoder.apply_fn({"params": self.context_encoder.params}, segments)
        return jax.lax.stop_gradient(z)

    def _qc(self, obs, actions, ctx):
        q1, q2 = self.cost_target_critic.apply_fn({"params": self.cost_target_critic.params}, _cat(obs, ctx), actions)
        return jnp.maximum(q1, q2)

    def _vc(self, obs, ctx):
        return self.cost_value.apply_fn({"params": self.cost_value.params}, _cat(obs, ctx))

    # -------------------------------------------------------- bcrl_det updates
    def update_cost(agent, batch, ctx):
        obs_ctx, nobs_ctx = _cat(batch["observations"], ctx), _cat(batch["next_observations"], ctx)

        def vc_loss_fn(params):
            q1, q2 = agent.cost_target_critic.apply_fn({"params": agent.cost_target_critic.params}, obs_ctx, batch["actions"])
            q = jax.lax.stop_gradient(jnp.maximum(q1, q2))
            v = agent.cost_value.apply_fn({"params": params}, obs_ctx)
            loss = expectile_loss(q - v, agent.cost_expectile).mean()
            return loss, {"cost_value_loss": loss, "mean_value_cost": v.mean(), "mean_q_cost": q.mean()}

        vc_grads, vc_info = jax.grad(vc_loss_fn, has_aux=True)(agent.cost_value.params)
        cost_value = agent.cost_value.apply_gradients(grads=vc_grads)

        next_v = cost_value.apply_fn({"params": cost_value.params}, nobs_ctx)
        target_q = batch["cost_raw"] + agent.cost_discount * batch["masks"] * next_v

        def qc_loss_fn(params):
            q1, q2 = agent.cost_critic.apply_fn({"params": params}, obs_ctx, batch["actions"])
            loss = (((target_q - q1) ** 2 + (target_q - q2) ** 2) / 2).mean()
            return loss, {"cost_critic_loss": loss}

        qc_grads, qc_info = jax.grad(qc_loss_fn, has_aux=True)(agent.cost_critic.params)
        cost_critic = agent.cost_critic.apply_gradients(grads=qc_grads)
        cost_target_params = optax.incremental_update(cost_critic.params, agent.cost_target_critic.params, agent.tau)
        cost_target_critic = agent.cost_target_critic.replace(params=cost_target_params)

        new_agent = agent.replace(cost_value=cost_value, cost_critic=cost_critic, cost_target_critic=cost_target_critic)
        return new_agent, {**vc_info, **qc_info}

    def update_critic(agent, batch, ctx, key):
        obs_ctx, nobs_ctx = _cat(batch["observations"], ctx), _cat(batch["next_observations"], ctx)

        qc1, qc2 = agent.cost_target_critic.apply_fn({"params": agent.cost_target_critic.params}, obs_ctx, batch["actions"])
        action_cost = jnp.maximum(qc1, qc2)
        next_state_cost = agent.cost_value.apply_fn({"params": agent.cost_value.params}, nobs_ctx)
        next_state_cost = jnp.maximum(next_state_cost, (action_cost - batch["cost_raw"]) / agent.cost_discount)
        next_state_cost = jnp.clip(next_state_cost, 0.0)

        deltabar_target = _sample_deltabar(key, next_state_cost.shape, next_state_cost, agent.delta_max)
        deltabar = batch["cost_raw"] + deltabar_target * agent.cost_discount

        next_v = agent.value.apply_fn({"params": agent.value.params}, nobs_ctx, deltabar_target[:, None])
        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def qr_loss_fn(params):
            q1, q2 = agent.critic.apply_fn({"params": params}, obs_ctx, batch["actions"], deltabar[:, None])
            loss = (((target_q - q1) ** 2 + (target_q - q2) ** 2) / 2).mean()
            return loss, {"critic_loss": loss}

        qr_grads, qr_info = jax.grad(qr_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=qr_grads)
        target_params = optax.incremental_update(critic.params, agent.target_critic.params, agent.tau)
        target_critic = agent.target_critic.replace(params=target_params)

        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, qr_info

    def update_value(agent, batch, ctx, key):
        obs_ctx = _cat(batch["observations"], ctx)
        qc1, qc2 = agent.cost_target_critic.apply_fn({"params": agent.cost_target_critic.params}, obs_ctx, batch["actions"])
        action_cost = jnp.maximum(qc1, qc2)
        deltabar = _sample_deltabar(key, action_cost.shape, action_cost, agent.delta_max)

        q1, q2 = agent.target_critic.apply_fn({"params": agent.target_critic.params}, obs_ctx, batch["actions"], deltabar[:, None])
        q = jax.lax.stop_gradient(jnp.minimum(q1, q2))

        def vr_loss_fn(params):
            v = agent.value.apply_fn({"params": params}, obs_ctx, deltabar[:, None])
            loss = expectile_loss(q - v, agent.expectile).mean()
            return loss, {"value_loss": loss, "mean_value": v.mean()}

        vr_grads, vr_info = jax.grad(vr_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=vr_grads)
        return agent.replace(value=value), vr_info

    def update_actor(agent, batch, ctx, key):
        obs_ctx = _cat(batch["observations"], ctx)
        qc1, qc2 = agent.cost_target_critic.apply_fn({"params": agent.cost_target_critic.params}, obs_ctx, batch["actions"])
        action_cost = jnp.maximum(qc1, qc2)
        state_cost = agent.cost_value.apply_fn({"params": agent.cost_value.params}, obs_ctx)
        deltabar = _sample_deltabar(key, action_cost.shape, action_cost, agent.delta_max)

        v = agent.value.apply_fn({"params": agent.value.params}, obs_ctx, deltabar[:, None])
        q1, q2 = agent.critic.apply_fn({"params": agent.critic.params}, obs_ctx, batch["actions"], deltabar[:, None])
        q = jnp.minimum(q1, q2)

        weight_a = jnp.clip(jnp.exp((q - v) * agent.beta), 0, 100)
        weight_s = jnp.clip(jnp.exp((state_cost - action_cost) * agent.beta), 0, 100)

        def actor_loss_fn(params):
            dist, dist_safe = agent.actor.apply_fn({"params": params}, obs_ctx, deltabar[:, None])
            log_probs = dist.log_prob(batch["actions"])
            log_probs_safe = dist_safe.log_prob(batch["actions"])
            loss = -(weight_a * log_probs).mean() - (weight_s * log_probs_safe).mean()
            return loss, {
                "actor_loss": loss, "mean_advantage": (q - v).mean(),
                "mean_safety_advantage": (state_cost - action_cost).mean(),
            }

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params)
        actor = agent.actor.apply_gradients(grads=grads)
        return agent.replace(actor=actor), info

    # ------------------------------------------------------------------ update
    @jax.jit
    def update(self, source_batch, target_batch, ctx_s, ctx_t):
        """One pooled bcrl_det step: (Q_C, V_C) then the budget-conditioned
        (Q_r, V_r, pi) on the D_s u D_t pool, each transition context-conditioned
        on its own regime's inferred zeta (source: zeta_s, target: zeta_t).
        `target_batch = ctx_t = None` drops the target side: the general policy
        (train_general.py), trained on the source regimes alone."""
        rng, k_cr, k_v, k_a = jax.random.split(self.rng, 4)
        new_agent = self.replace(rng=rng)

        if ctx_s.ndim == 1:
            ctx_s = jnp.broadcast_to(ctx_s, (source_batch["observations"].shape[0], self.context_dim))

        keys = ("observations", "actions", "rewards", "next_observations", "masks", "cost_raw")
        if target_batch is None:
            pooled = {k: source_batch[k] for k in keys}
            ctx = ctx_s
        else:
            if ctx_t.ndim == 1:
                ctx_t = jnp.broadcast_to(ctx_t, (target_batch["observations"].shape[0], self.context_dim))
            pooled = {k: jnp.concatenate([source_batch[k], target_batch[k]], axis=0) for k in keys}
            ctx = jnp.concatenate([ctx_s, ctx_t], axis=0)

        new_agent, c_info = new_agent.update_cost(pooled, ctx)
        new_agent, cr_info = new_agent.update_critic(pooled, ctx, k_cr)
        new_agent, v_info = new_agent.update_value(pooled, ctx, k_v)
        new_agent, a_info = new_agent.update_actor(pooled, ctx, k_a)

        return new_agent, {**c_info, **cr_info, **v_info, **a_info}

    # -------------------------------------------------------------- deployment
    def set_context(self, ctx_vec):
        """Burn-in freeze (Prop. `prop:context-switching`(1))."""
        return self.replace(ctx=jnp.asarray(ctx_vec, dtype=jnp.float32))

    def reset_budget(self, obs0, delta_init):
        """delta_0 = the soft-constraint budget under test (`bcrl_det`'s
        `cost_limits`, tracked online from here on)."""
        return self.replace(delta_hat=float(delta_init))

    @jax.jit
    def step_budget(self, obs, action, next_obs):
        """g(s,a,s',delta) = V_C(s') + (delta - Q_C(s,a)) / gamma."""
        ctx = jnp.expand_dims(self.ctx, 0)
        qc = self._qc(jnp.expand_dims(obs, 0), jnp.expand_dims(action, 0), ctx)[0]
        next_vc = self._vc(jnp.expand_dims(next_obs, 0), ctx)[0]
        return self.replace(delta_hat=next_vc + (self.delta_hat - qc) / self.cost_discount)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        """`bcrl_det.IQL.get_action`: sample from pi_theta(.|s,zeta,delta) when
        the remaining budget covers the state's baseline cost V_C(s,zeta), else
        fall back to the unconditioned safe actor pi_safe(.|s,zeta)."""
        rng, key = jax.random.split(self.rng, 2)
        observations = jax.device_put(observations)
        obs = jnp.expand_dims(observations, axis=0)
        ctx = jnp.expand_dims(self.ctx, 0)
        obs_ctx = _cat(obs, ctx)

        state_cost = self.cost_value.apply_fn({"params": self.cost_value.params}, obs_ctx)
        deltabar = jnp.clip(jnp.asarray(self.delta_hat, dtype=jnp.float32), 0.0)
        delta_in = jnp.reshape(deltabar, (1, 1))

        dist, dist_safe = self.actor.apply_fn({"params": self.actor.params}, obs_ctx, delta_in, temperature=0.0)
        r_action = dist.sample(seed=key)
        safe_action = dist_safe.sample(seed=key)
        action = jnp.where(deltabar > state_cost, r_action, safe_action)
        return action.squeeze(), self.replace(rng=rng)

    def save(self, modeldir, save_time):
        file_name = "model" + str(save_time) + ".pickle"
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), "wb"))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, "rb"))
        return flax.serialization.from_state_dict(target=self, state=pkl_file)
