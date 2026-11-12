"""Part III (Sec. `sec:pooled`): pooled multi-source training with an inferred
constraint context, on top of the budget-conditioned soft-constraint pipeline of
Sec. `sec:soft-algo`.

Relative to `CBFTransfer` this class adds, all conditioned on a constraint
context `zeta` that a frozen `q_xi` infers from a cost-annotated segment:

  * an *additive* cost critic pair  Q_C^rho(z, zeta), V_C^sigma(s, zeta)  - the
    `Psi_C(h,v) = h + gamma v` instantiation of the operator, fit at both z and
    its forward projection f_eta(z) (Lemma `lem:cost-consistency` /
    `lem:ctx-cost-consistency`), with an expectile tau_C <= 0.5 on V_C;

  * context-conditioning of the reachability barrier  Q_h^phi(z, zeta),
    V_h^nu(s, zeta)  (Sec. `sec:pooled-algo` step 3);

  * a budget-conditioned reward critic / value / policy - Q_r^omega(s, a, delta),
    V_r^psi(s, delta), pi_theta(a | s, delta) - trained with the
    budget-generalised target of eq. (17); the reward side is conditioned on
    `delta` but *not* on `zeta` (Sec. `sec:pooled-practical`, Stage 5);

  * the soft budget-tracking rule  g(s,a,s',delta) = V_C(s') + (delta-Q_C(z))/gamma
    (eq. 16), a per-minibatch budget sample  delta ~ U[Q_C(z,zeta), delta_max],
    and a deployment path with a burn-in-frozen context and the deflated tests of
    Corollary `cor:pool-margin`.

Setting delta = 0 and a single regime recovers `CBFTransfer` (Thm. `thm:nesting`).

This is the `multi_src/2` port of `multi_src/1/models/agents/cbf/cbf_pooled_g.py`:
the additive-operator budget bookkeeping (eq. 15/16/17) is kept self-contained
here (as in `multi_src/1`) rather than reusing `CBFTransfer`'s bcrl-style
`sample_deltabar` conditioning, so the pooled certificate matches FPCBF_multi.tex
exactly.  `safe_score_model` / `cost_discount` / `budget_max` / `cost_temperature`
are only carried to satisfy the shared `CBF` dataclass and are unused here.
"""

from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from models.agents.cbf.cbf_g import CBF, expectile_loss
from models.agents.cbf.cbf_transfer_g import CBFTransfer
from models.agents.cbf.flow_transfer import _integrate_flow
from models.data.dataset import DatasetDict
from models.networks import Ensemble, GaussianPolicy, MLP, StateActionValue, StateValue, get_weight_decay_mask
from models.networks import cosine_beta_schedule, vp_beta_schedule
from models.networks.transfer_nets import AutoEncoder


def _cat(*xs):
    return jnp.concatenate(xs, axis=-1)


def _aug(obs, delta):
    """Append the running budget as an extra observation coordinate."""
    return jnp.concatenate([obs, jnp.reshape(delta, (-1, 1))], axis=-1)


class CBFPooled(CBFTransfer):
    # `cost_critic` / `cost_target_critic` / `cost_value` / `delta_hat` are
    # inherited from CBF (re-initialised over the latent+context input in
    # create()); they are NOT redeclared here -- adding a default to an inherited
    # no-default field would break the dataclass field ordering.  Only genuinely
    # new fields are declared, all with defaults.

    # --- frozen constraint-context encoder q_xi --------------------------------
    context_encoder: TrainState = None
    context_dim: int = struct.field(pytree_node=False, default=8)
    seg_m: int = struct.field(pytree_node=False, default=16)
    # --- soft-constraint hyper-parameters -------------------------------------
    cost_tau_c: float = 0.3
    delta_max: float = 50.0
    # --- deployment deflation margins (Corollary `cor:pool-margin`) -----------
    eps_c_star: float = 0.0     # epsilon_c^*  (best-matching-context alignment residual)
    eps_h: float = 0.0         # epsilon_h^*  (hard-barrier alignment residual)
    lc_epsz: float = 0.0      # L_c * epsilon_zeta  (context-inference displacement, budget critic)
    eps_opt: float = 0.0       # critic optimisation error
    # --- runtime state -------------------------------------------------------
    ctx: jnp.ndarray = None
    vc_mu0: float = 0.0        # E_{mu0}[V_C(s)] estimate for eq. (15) re-centring

    # ---------------------------------------------------------------- factory
    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        autoencoder: TrainState,
        velocity: TrainState,
        context_encoder: TrainState,
        latent_dim: int,
        context_dim: int,
        seg_m: int = 16,
        flow_steps: int = 10,
        lambda_t: float = 1.0,
        cost_tau_c: float = 0.3,
        delta_max: float = 50.0,
        eps_c_star: float = 0.0,
        eps_h: float = 0.0,
        lc_epsz: float = 0.0,
        eps_opt: float = 0.0,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        cbf_lr: float = 1e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        reward_tau: float = 0.8,
        num_qs: int = 2,
        actor_weight_decay: Optional[float] = None,
        value_layer_norm: bool = False,
        critic_layer_norm: bool = True,
        reward_temperature: float = 3.0,
        cost_ub: float = 150.0,
        N: int = 64,
        decay_steps: Optional[int] = int(2e6),
        cost_tau: float = 0.2,
        r_min: float = -1.0,
        mode: int = 1,
        actor_tau: float = 0.001,
        cost_limit: float = 10,
        T: int = 5,
        time_dim: int = 64,
        M: int = 0,
        ddpm_temperature: float = 0.1,
        clip_sampler: bool = True,
        beta_schedule: str = "linear",
        tanh_scale: float = 5,
        # carried only for the shared CBF dataclass (unused by the pooled path)
        cost_discount: float = 0.99,
        cost_expectile: float = 0.3,
        budget_max: Optional[float] = None,
        cost_temperature: float = 5.0,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, safe_actor_key, critic_key, value_key, safe_critic_key, safe_value_key = jax.random.split(rng, 7)
        rng, cost_critic_key, cost_value_key = jax.random.split(rng, 3)

        observations = jnp.expand_dims(observation_space.sample(), axis=0)
        actions = jnp.expand_dims(action_space.sample(), axis=0)
        action_dim = action_space.shape[0]
        obs_dim = observations.shape[-1]
        if budget_max is None:
            budget_max = delta_max

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        obs_aug = _aug(observations, jnp.zeros((1,)))          # (1, obs_dim + 1)
        z_ctx = jnp.zeros((1, latent_dim + context_dim))
        obs_ctx = jnp.zeros((1, obs_dim + context_dim))

        # --- budget-conditioned policy pi_theta(a | s, delta) -----------------
        actor_def = GaussianPolicy(hidden_dims=actor_hidden_dims, action_dim=action_dim)
        actor_params = FrozenDict(actor_def.init(actor_key, obs_aug)["params"])
        score_model = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adamw(
                learning_rate=actor_lr,
                weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                mask=get_weight_decay_mask,
            ),
        )

        # --- unconditioned safe fallback actor (carried for the CBF dataclass) ---
        safe_actor_params = FrozenDict(actor_def.init(safe_actor_key, observations)["params"])
        safe_score_model = TrainState.create(
            apply_fn=actor_def.apply,
            params=safe_actor_params,
            tx=optax.adamw(
                learning_rate=actor_lr,
                weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                mask=get_weight_decay_mask,
            ),
        )

        # --- budget-conditioned reward critic Q_r(s, a, delta) ----------------
        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=critic_layer_norm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = FrozenDict(critic_def.init(critic_key, obs_aug, actions)["params"])
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(learning_rate=critic_lr))
        target_critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        # --- context-conditioned reachability barrier Q_h^phi(z, zeta) --------
        safe_critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=critic_layer_norm)
        safe_critic_cls = partial(StateValue, base_cls=safe_critic_base_cls)
        safe_critic_def = Ensemble(safe_critic_cls, num=num_qs)
        safe_critic_params = FrozenDict(safe_critic_def.init(safe_critic_key, z_ctx)["params"])
        safe_critic = TrainState.create(apply_fn=safe_critic_def.apply, params=safe_critic_params, tx=optax.adam(learning_rate=cbf_lr))
        safe_target_critic = TrainState.create(
            apply_fn=safe_critic_def.apply, params=safe_critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        # --- additive cost critic Q_C^rho(z, zeta) ----------------------------
        cost_critic_def = Ensemble(safe_critic_cls, num=num_qs)
        cost_critic_params = FrozenDict(cost_critic_def.init(cost_critic_key, z_ctx)["params"])
        cost_critic = TrainState.create(apply_fn=cost_critic_def.apply, params=cost_critic_params, tx=optax.adam(learning_rate=cbf_lr))
        cost_target_critic = TrainState.create(
            apply_fn=cost_critic_def.apply, params=cost_critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        # --- values --------------------------------------------------------
        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        value_params = FrozenDict(value_def.init(value_key, obs_aug)["params"])          # V_r(s, delta)
        value = TrainState.create(apply_fn=value_def.apply, params=value_params, tx=optax.adam(learning_rate=value_lr))

        safe_value_def = StateValue(base_cls=value_base_cls)
        safe_value_params = FrozenDict(safe_value_def.init(safe_value_key, obs_ctx)["params"])   # V_h(s, zeta)
        safe_value = TrainState.create(apply_fn=safe_value_def.apply, params=safe_value_params, tx=optax.adam(learning_rate=value_lr))

        cost_value_def = StateValue(base_cls=value_base_cls)
        cost_value_params = FrozenDict(cost_value_def.init(cost_value_key, obs_ctx)["params"])   # V_C(s, zeta)
        cost_value = TrainState.create(apply_fn=cost_value_def.apply, params=cost_value_params, tx=optax.adam(learning_rate=value_lr))

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")
        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[: i + 1]) for i in range(T)])

        return cls(
            actor=None,
            score_model=score_model,
            safe_score_model=safe_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            tau=tau,
            discount=discount,
            rng=rng,
            action_dim=action_dim,
            N=N,
            reward_tau=reward_tau,
            reward_temperature=reward_temperature,
            cost_tau=cost_tau,
            cost_ub=cost_ub,
            r_min=r_min,
            mode=mode,
            actor_tau=actor_tau,
            cost_discount=cost_discount,
            cost_expectile=cost_expectile,
            budget_max=budget_max,
            cost_temperature=cost_temperature,
            T=T,
            M=M,
            ddpm_temperature=ddpm_temperature,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hat,
            clip_sampler=clip_sampler,
            tanh_scale=tanh_scale,
            beta_schedule=beta_schedule,
            autoencoder=autoencoder,
            velocity=velocity,
            flow_steps=flow_steps,
            latent_dim=latent_dim,
            lambda_t=lambda_t,
            cost_critic=cost_critic,
            cost_target_critic=cost_target_critic,
            cost_value=cost_value,
            context_encoder=context_encoder,
            context_dim=context_dim,
            seg_m=seg_m,
            cost_tau_c=cost_tau_c,
            delta_max=delta_max,
            eps_c_star=eps_c_star,
            eps_h=eps_h,
            lc_epsz=lc_epsz,
            eps_opt=eps_opt,
            delta_hat=0.0,
            ctx=jnp.zeros((context_dim,)),
            vc_mu0=0.0,
        )

    # ----------------------------------------------------------------- helpers
    def infer_context(self, segments):
        """zeta_hat = q_xi(sigma).  Frozen: no gradient ever flows into q_xi."""
        z = self.context_encoder.apply_fn({"params": self.context_encoder.params}, segments)
        return jax.lax.stop_gradient(z)

    def _qc(self, z, ctx, target=False):
        params = self.cost_target_critic.params if target else self.cost_critic.params
        return self.cost_critic.apply_fn({"params": params}, _cat(z, ctx)).min(axis=0)

    def _qh(self, z, ctx, target=False):
        params = self.safe_target_critic.params if target else self.safe_critic.params
        return self.safe_critic.apply_fn({"params": params}, _cat(z, ctx)).max(axis=0)

    def _vc(self, obs, ctx):
        return self.cost_value.apply_fn({"params": self.cost_value.params}, _cat(obs, ctx))

    def _sample_budget(self, key, z, ctx):
        """delta ~ U[Q_C(z, zeta), delta_max]  (Sec. `sec:soft-algo`, Step 3)."""
        qc = self._qc(z, ctx)
        u = jax.random.uniform(key, qc.shape)
        span = jnp.maximum(self.delta_max - qc, 0.0)
        return qc + u * span, qc

    def _budget_next(self, delta, qc_z, next_vc):
        """g(s,a,s',delta) = V_C(s') + (delta - Q_C(z)) / gamma   (eq. 16)."""
        return next_vc + (delta - qc_z) / self.discount

    @jax.jit
    def barrier_values(self, observations):
        ctx = jnp.broadcast_to(self.ctx, (observations.shape[0], self.context_dim))
        return self.safe_value.apply_fn({"params": self.safe_value.params}, _cat(observations, ctx))

    # ------------------------------------------------------------- cost critic
    def update_cost(agent, source_batch, target_batch, ctx_s, ctx_t):
        """Additive-operator fit of (Q_C, V_C), context-conditioned, dual-fit at z
        and f_eta(z) on the source (Lemma `lem:ctx-cost-consistency`)."""
        z_s = jax.lax.stop_gradient(agent._encode(source_batch["observations"], source_batch["actions"]))
        z_t = jax.lax.stop_gradient(agent._encode(target_batch["observations"], target_batch["actions"]))
        z_s_proj = jax.lax.stop_gradient(_integrate_flow(agent.velocity.apply_fn, agent.velocity.params, z_s, agent.flow_steps))

        c_s = source_batch["cost_raw"]
        c_t = target_batch["cost_raw"]

        z_pool = jnp.concatenate([z_s, z_t], axis=0)
        ctx_pool = jnp.concatenate([ctx_s, ctx_t], axis=0)
        obs_pool = jnp.concatenate([source_batch["observations"], target_batch["observations"]], axis=0)

        qC_tgt_pool = agent.cost_critic.apply_fn({"params": agent.cost_target_critic.params}, _cat(z_pool, ctx_pool)).min(axis=0)

        def Vc_loss(params):
            vc = agent.cost_value.apply_fn({"params": params}, _cat(obs_pool, ctx_pool))
            loss = expectile_loss(qC_tgt_pool - vc, agent.cost_tau_c).mean()   # tau_C <= 0.5 -> min-biased
            return loss, {"vc_loss": loss, "v_c": vc.mean()}

        vc_grads, vc_info = jax.grad(Vc_loss, has_aux=True)(agent.cost_value.params)
        cost_value = agent.cost_value.apply_gradients(grads=vc_grads)

        next_vc_s = cost_value.apply_fn({"params": cost_value.params}, _cat(source_batch["next_observations"], ctx_s))
        next_vc_t = cost_value.apply_fn({"params": cost_value.params}, _cat(target_batch["next_observations"], ctx_t))
        y_C_s = c_s + agent.discount * source_batch["masks"] * next_vc_s
        y_C_t = c_t + agent.discount * target_batch["masks"] * next_vc_t

        def Qc_loss(params):
            q_s = agent.cost_critic.apply_fn({"params": params}, _cat(z_s, ctx_s))
            q_s_proj = agent.cost_critic.apply_fn({"params": params}, _cat(z_s_proj, ctx_s))
            q_t = agent.cost_critic.apply_fn({"params": params}, _cat(z_t, ctx_t))
            source_loss = (((q_s - y_C_s) ** 2) + ((q_s_proj - y_C_s) ** 2)).mean()
            target_loss = ((q_t - y_C_t) ** 2).mean()
            loss = source_loss + agent.lambda_t * target_loss
            return loss, {
                "qc_loss": loss,
                "qc_source_loss": source_loss,
                "qc_target_loss": target_loss,
                "q_c": q_s.mean(),
            }

        qc_grads, qc_info = jax.grad(Qc_loss, has_aux=True)(agent.cost_critic.params)
        cost_critic = agent.cost_critic.apply_gradients(grads=qc_grads)
        cost_target_params = optax.incremental_update(cost_critic.params, agent.cost_target_critic.params, agent.tau)
        cost_target_critic = agent.cost_target_critic.replace(params=cost_target_params)

        new_agent = agent.replace(cost_value=cost_value, cost_critic=cost_critic, cost_target_critic=cost_target_critic)
        return new_agent, {**vc_info, **qc_info}

    # ---------------------------------------------------- context-cond. barrier
    def update_h(agent, source_batch, target_batch, ctx_s, ctx_t):
        z_s = jax.lax.stop_gradient(agent._encode(source_batch["observations"], source_batch["actions"]))
        z_t = jax.lax.stop_gradient(agent._encode(target_batch["observations"], target_batch["actions"]))
        z_s_proj = jax.lax.stop_gradient(_integrate_flow(agent.velocity.apply_fn, agent.velocity.params, z_s, agent.flow_steps))

        h_s = source_batch["costs"]
        h_t = target_batch["costs"]

        z_pool = jnp.concatenate([z_s, z_t], axis=0)
        ctx_pool = jnp.concatenate([ctx_s, ctx_t], axis=0)
        obs_pool = jnp.concatenate([source_batch["observations"], target_batch["observations"]], axis=0)

        qc_pool = agent.safe_critic.apply_fn({"params": agent.safe_target_critic.params}, _cat(z_pool, ctx_pool)).max(axis=0)

        def Vh_loss(params):
            vh = agent.safe_value.apply_fn({"params": params}, _cat(obs_pool, ctx_pool))
            loss = expectile_loss(qc_pool - vh, agent.cost_tau).mean()
            return loss, {"vh_loss": loss, "v_h": vh.mean()}

        vh_grads, vh_info = jax.grad(Vh_loss, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=vh_grads)

        next_vh_s = safe_value.apply_fn({"params": safe_value.params}, _cat(source_batch["next_observations"], ctx_s))
        next_vh_t = safe_value.apply_fn({"params": safe_value.params}, _cat(target_batch["next_observations"], ctx_t))
        y_h_s = (1 - agent.discount) * h_s + agent.discount * jnp.maximum(h_s, next_vh_s)
        y_h_t = (1 - agent.discount) * h_t + agent.discount * jnp.maximum(h_t, next_vh_t)

        def Qh_loss(params):
            qhs_s = agent.safe_critic.apply_fn({"params": params}, _cat(z_s, ctx_s))
            qhs_s_proj = agent.safe_critic.apply_fn({"params": params}, _cat(z_s_proj, ctx_s))
            qhs_t = agent.safe_critic.apply_fn({"params": params}, _cat(z_t, ctx_t))
            source_loss = (jnp.abs(qhs_s - y_h_s) + jnp.abs(qhs_s_proj - y_h_s)).mean()
            target_loss = jnp.abs(qhs_t - y_h_t).mean()
            loss = source_loss + agent.lambda_t * target_loss
            return loss, {"qh_loss": loss, "qh_source_loss": source_loss, "qh_target_loss": target_loss, "q_h": qhs_s.mean()}

        qh_grads, qh_info = jax.grad(Qh_loss, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=qh_grads)
        safe_target_params = optax.incremental_update(safe_critic.params, agent.safe_target_critic.params, agent.tau)
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_params)

        new_agent = agent.replace(safe_value=safe_value, safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, {**vh_info, **qh_info}

    # ----------------------------------------------- budget-conditioned reward
    def update_r(agent, batch, ctx_t_vec, key):
        B = batch["observations"].shape[0]
        ctx_t = jnp.broadcast_to(ctx_t_vec, (B, agent.context_dim))
        z = jax.lax.stop_gradient(agent._encode(batch["observations"], batch["actions"]))

        delta, qc_z = agent._sample_budget(key, z, ctx_t)
        qh_z = agent._qh(z, ctx_t)
        next_vc = agent._vc(batch["next_observations"], ctx_t)
        delta_next = agent._budget_next(delta, qc_z, next_vc)

        obs_d = _aug(batch["observations"], delta)
        nobs_d = _aug(batch["next_observations"], delta_next)

        qs = agent.target_critic.apply_fn({"params": agent.target_critic.params}, obs_d, batch["actions"]).min(axis=0)

        def Vr_loss(params):
            v = agent.value.apply_fn({"params": params}, obs_d)
            loss = expectile_loss(qs - v, agent.reward_tau).mean()
            return loss, {"v_r_loss": loss, "v_r": v.mean()}

        vr_grads, vr_info = jax.grad(Vr_loss, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=vr_grads)

        next_vr = value.apply_fn({"params": value.params}, nobs_d)
        feasible = (qc_z <= delta) | (qh_z <= 0.0)                       # eq. (17) condition
        safe_target = batch["rewards"] + agent.discount * batch["masks"] * next_vr
        unsafe_target = agent.r_min / (1 - agent.discount) - (qc_z - delta)
        target_q = jnp.where(feasible, safe_target, unsafe_target)

        def Qr_loss(params):
            qr = agent.critic.apply_fn({"params": params}, obs_d, batch["actions"])
            loss = ((qr - target_q) ** 2).mean()
            return loss, {"q_r_loss": loss, "q_r": qr.mean(), "feasible_frac": feasible.mean()}

        qr_grads, qr_info = jax.grad(Qr_loss, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=qr_grads)
        target_critic_params = optax.incremental_update(critic.params, agent.target_critic.params, agent.tau)
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(value=value, critic=critic, target_critic=target_critic)
        return new_agent, {**vr_info, **qr_info}

    def update_actor(agent, batch, ctx_t_vec, key):
        B = batch["observations"].shape[0]
        ctx_t = jnp.broadcast_to(ctx_t_vec, (B, agent.context_dim))
        z = jax.lax.stop_gradient(agent._encode(batch["observations"], batch["actions"]))
        delta, _ = agent._sample_budget(key, z, ctx_t)
        obs_d = _aug(batch["observations"], delta)

        q = agent.target_critic.apply_fn({"params": agent.target_critic.params}, obs_d, batch["actions"]).min(axis=0)
        v = agent.value.apply_fn({"params": agent.value.params}, obs_d)
        weights = jnp.clip(jnp.exp((q - v) * agent.reward_temperature), 0, 100)

        def actor_loss_fn(params):
            dist = agent.score_model.apply_fn({"params": params}, obs_d)
            log_probs = dist.log_prob(batch["actions"])
            loss = -(log_probs * weights).mean()
            return loss, {"weights": weights.mean(), "log_probs": log_probs.mean()}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        return agent.replace(score_model=score_model), info

    # ------------------------------------------------------------------ update
    @jax.jit
    def update(self, source_batch: DatasetDict, target_batch: DatasetDict, ctx_s: jnp.ndarray, ctx_t: jnp.ndarray):
        """One pooled step: (Q_h, V_h) and (Q_C, V_C) fit context-conditioned on
        source (direct + forward-projected) and target; then the budget-conditioned
        Q_r, V_r and pi_theta on the D_s u D_t pool, feasibility gated at zeta_t."""
        rng, k_r, k_a = jax.random.split(self.rng, 3)
        new_agent = self.replace(rng=rng)

        if ctx_t.ndim == 1:
            ctx_t = jnp.broadcast_to(ctx_t, (target_batch["observations"].shape[0], self.context_dim))
        if ctx_s.ndim == 1:
            ctx_s = jnp.broadcast_to(ctx_s, (source_batch["observations"].shape[0], self.context_dim))

        new_agent, h_info = new_agent.update_h(source_batch, target_batch, ctx_s, ctx_t)
        new_agent, c_info = new_agent.update_cost(source_batch, target_batch, ctx_s, ctx_t)

        pooled = {
            k: jnp.concatenate([source_batch[k], target_batch[k]], axis=0)
            for k in ("observations", "actions", "rewards", "next_observations", "masks", "cost_raw")
        }
        ctx_t_vec = jnp.mean(ctx_t, axis=0)
        new_agent, r_info = new_agent.update_r(pooled, ctx_t_vec, k_r)
        new_agent, a_info = new_agent.update_actor(pooled, ctx_t_vec, k_a)

        return new_agent, {**h_info, **c_info, **r_info, **a_info}

    # -------------------------------------------------------------- deployment
    def set_context(self, ctx_vec):
        """Burn-in freeze (Prop. `prop:context-switching`(1))."""
        return self.replace(ctx=jnp.asarray(ctx_vec, dtype=jnp.float32))

    def set_vc_mu0(self, value):
        return self.replace(vc_mu0=float(value))

    def reset_budget(self, obs0, delta_init):
        """delta_0 = V_C(s_0) + delta_init - E_{mu0}[V_C(s)]   (eq. 15)."""
        vc0 = self._vc(jnp.expand_dims(jnp.asarray(obs0, jnp.float32), 0),
                       jnp.expand_dims(self.ctx, 0))[0]
        delta0 = float(vc0) + float(delta_init) - self.vc_mu0
        return self.replace(delta_hat=delta0)

    @jax.jit
    def step_budget(self, obs, action, next_obs):
        """Advance the running budget by g(...) after an environment step."""
        z = self._encode(jnp.expand_dims(obs, 0), jnp.expand_dims(action, 0))
        ctx = jnp.expand_dims(self.ctx, 0)
        qc = self._qc(z, ctx)[0]
        next_vc = self._vc(jnp.expand_dims(next_obs, 0), ctx)[0]
        return self.replace(delta_hat=next_vc + (self.delta_hat - qc) / self.discount)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        """Sample N actions from pi_theta(. | s, delta); keep those passing the
        deflated tests of Corollary `cor:pool-margin`; take the reward-maximising
        survivor, else the cost-greedy fallback."""
        rng, key = jax.random.split(self.rng, 2)
        observations = jax.device_put(observations)
        obs_b = jnp.expand_dims(observations, axis=0).repeat(self.N, axis=0)
        delta_b = jnp.full((self.N,), self.delta_hat)
        ctx_b = jnp.broadcast_to(self.ctx, (self.N, self.context_dim))

        dist = self.score_model.apply_fn({"params": self.score_model.params}, _aug(obs_b, delta_b), temperature=0.1)
        actions = dist.sample(seed=key)

        z = self._encode(obs_b, actions)
        qc = self._qc(z, ctx_b)
        qh = self._qh(z, ctx_b)

        delta_eff = self.delta_hat - (self.eps_c_star + self.lc_epsz) / (1 - self.discount) - self.eps_opt
        hard_thr = -(self.eps_h + self.lc_epsz + self.eps_opt)
        feasible = (qc <= delta_eff) | (qh <= hard_thr)

        qr = self.critic.apply_fn({"params": self.critic.params}, _aug(obs_b, delta_b), actions).min(axis=0)
        masked_qr = jnp.where(feasible, qr, -1e10)
        idx = jnp.where(feasible.any(), jnp.argmax(masked_qr), jnp.argmin(qc))
        action = actions[idx]
        return action.squeeze(), self.replace(rng=rng)
