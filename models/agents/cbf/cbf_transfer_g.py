from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union , Any

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax import struct
from flax.training.train_state import TrainState

from models.agents.cbf.cbf_g import CBF, expectile_loss, sample_deltabar
from models.data.dataset import DatasetDict
from models.networks import MLP, Ensemble, StateActionValue, StateValue, get_weight_decay_mask
from models.networks import GaussianPolicy
from models.networks import DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler, vp_beta_schedule
from models.networks.transfer_nets import AutoEncoder
from models.agents.cbf.flow_transfer import _integrate_flow


@partial(jax.jit, static_argnames=("safe_critic_fn",))
def compute_safe_q_latent(safe_critic_fn, safe_critic_params, z):
    safe_q_values = safe_critic_fn({"params": safe_critic_params}, z)
    return safe_q_values.max(axis=0)


class CBFTransfer(CBF):
    """Stage 3: Q_h^phi is defined over the shared latent Z instead of raw S x A,
    and is fit jointly on source, its forward-projected image, and real target
    data (the former Stage 4 inverse-flow projection is merged directly into
    training, so Q_h^phi is valid at both source and target latents by
    construction - no separate projection step is needed at eval time).

    Q_r, V_r and pi_theta still only ever see raw (s, a), but are now trained on
    data pooled across both domains (D_s u D_t): reward and dynamics are shared
    between source and target, so pooling introduces no relabeling risk.

    Non-zero soft-constraint budgets are handled exactly as in CBF/direct: a
    cumulative cost critic Q_C / V_C (trained on the pooled data) supplies the
    budget bookkeeping, and Q_r / V_r / pi_theta are conditioned on a sampled
    remaining budget (delta-bar).
    """

    autoencoder: TrainState
    velocity: TrainState
    flow_steps: int = struct.field(pytree_node=False)
    latent_dim: int = struct.field(pytree_node=False)
    lambda_t: float = 1.0

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        autoencoder: TrainState,
        velocity: TrainState,
        latent_dim: int,
        flow_steps: int = 10,
        lambda_t: float = 1.0,
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
        cost_discount: float = 0.99,
        cost_expectile: float = 0.3,
        budget_max: float = 100.0,
        cost_temperature: float = 5.0,
    ):
        rng = jax.random.PRNGKey(seed)
        (rng, actor_key, safe_actor_key, critic_key, value_key,
         safe_critic_key, safe_value_key, cost_critic_key, cost_value_key) = jax.random.split(rng, 9)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        preprocess_time_cls = partial(FourierFeatures, output_size=time_dim, learnable=True)
        cond_model_cls = partial(MLP, hidden_dims=(128, 128), activations=nn.relu, activate_final=False)
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        base_model_cls = partial(MLP, hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]), activations=nn.relu, activate_final=False)
        actor_def = GaussianPolicy(hidden_dims=actor_hidden_dims, action_dim=action_dim)

        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis=0)
        budget_sample = jnp.zeros((1, 1))

        # ---- reward actor (budget-conditioned) + safe fallback actor ----
        actor_params = FrozenDict(actor_def.init(actor_key, observations, budget_sample)['params'])
        score_model = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adamw(
                learning_rate=actor_lr,
                weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                mask=get_weight_decay_mask,
            ),
        )
        safe_actor_params = FrozenDict(actor_def.init(safe_actor_key, observations)['params'])
        safe_score_model = TrainState.create(
            apply_fn=actor_def.apply,
            params=safe_actor_params,
            tx=optax.adamw(
                learning_rate=actor_lr,
                weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                mask=get_weight_decay_mask,
            ),
        )

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=critic_layer_norm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        # reward critic is budget-conditioned
        critic_params = FrozenDict(critic_def.init(critic_key, observations, actions, budget_sample)["params"])
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(learning_rate=critic_lr))
        target_critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        # Q_h^phi : Z -> R (ensemble), unlike CBF's safe_critic which reads raw (s, a).
        safe_critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=critic_layer_norm)
        safe_critic_cls = partial(StateValue, base_cls=safe_critic_base_cls)
        safe_critic_def = Ensemble(safe_critic_cls, num=num_qs)
        z_dummy = jnp.zeros((1, latent_dim))
        safe_critic_params = FrozenDict(safe_critic_def.init(safe_critic_key, z_dummy)["params"])
        safe_critic_optimiser = optax.adam(learning_rate=cbf_lr)
        safe_critic = TrainState.create(apply_fn=safe_critic_def.apply, params=safe_critic_params, tx=safe_critic_optimiser)
        safe_target_critic = TrainState.create(
            apply_fn=safe_critic_def.apply, params=safe_critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        # ---- cumulative cost critic Q_C / value V_C (raw space, for the soft budget) ----
        cost_critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        cost_critic_def = Ensemble(cost_critic_cls, num=num_qs)
        cost_critic_params = FrozenDict(cost_critic_def.init(cost_critic_key, observations, actions)["params"])
        cost_critic = TrainState.create(
            apply_fn=cost_critic_def.apply, params=cost_critic_params, tx=optax.adam(learning_rate=cbf_lr)
        )
        cost_target_critic = TrainState.create(
            apply_fn=cost_critic_def.apply, params=cost_critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        # reward value is budget-conditioned
        value_params = FrozenDict(value_def.init(value_key, observations, budget_sample)["params"])
        value_optimiser = optax.adam(learning_rate=value_lr)
        value = TrainState.create(apply_fn=value_def.apply, params=value_params, tx=value_optimiser)

        safe_value_def = StateValue(base_cls=value_base_cls)
        safe_value_params = FrozenDict(safe_value_def.init(safe_value_key, observations)["params"])
        safe_value = TrainState.create(apply_fn=safe_value_def.apply, params=safe_value_params, tx=value_optimiser)

        cost_value_def = StateValue(base_cls=value_base_cls)
        cost_value_params = FrozenDict(cost_value_def.init(cost_value_key, observations)["params"])
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
            cost_critic=cost_critic,
            cost_target_critic=cost_target_critic,
            cost_value=cost_value,
            delta_hat=jnp.zeros((1,)),
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
        )

    def _encode(self, observations, actions):
        return self.autoencoder.apply_fn({"params": self.autoencoder.params}, observations, actions, method=AutoEncoder.encode)

    def update_h(
        agent, source_batch: DatasetDict, target_batch: DatasetDict
    ) -> Tuple["CBFTransfer", Dict[str, float]]:
        """Stage 3: fit Q_h^phi on source (direct + forward-projected f_eta(z_s)) and on
        real target data simultaneously, weighting the target term by lambda_t. V_h^nu is
        regressed on the pool M_s u M_t (reward/dynamics-free, so pooling is safe here too)."""
        z_s = jax.lax.stop_gradient(agent._encode(source_batch["observations"], source_batch["actions"]))
        z_t = jax.lax.stop_gradient(agent._encode(target_batch["observations"], target_batch["actions"]))
        z_s_proj = jax.lax.stop_gradient(_integrate_flow(agent.velocity.apply_fn, agent.velocity.params, z_s, agent.flow_steps))

        h_s = source_batch["costs"]
        h_t = target_batch["costs"]

        z_pooled = jnp.concatenate([z_s, z_t], axis=0)
        obs_pooled = jnp.concatenate([source_batch["observations"], target_batch["observations"]], axis=0)
        qcs_pooled = agent.safe_target_critic.apply_fn({"params": agent.safe_target_critic.params}, z_pooled)
        qc_pooled = qcs_pooled.max(axis=0)

        def Vh_loss(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vh = agent.safe_value.apply_fn({"params": safe_value_params}, obs_pooled)
            loss = expectile_loss(qc_pooled - vh, agent.cost_tau).mean()
            return loss, {"vh_loss": loss, "v_h": vh.mean()}

        vh_grads, vh_info = jax.grad(Vh_loss, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=vh_grads)

        next_vh_s = safe_value.apply_fn({"params": safe_value.params}, source_batch["next_observations"])
        next_vh_t = safe_value.apply_fn({"params": safe_value.params}, target_batch["next_observations"])
        y_h_s = (1 - agent.discount) * h_s + agent.discount * jnp.maximum(h_s, next_vh_s)
        y_h_t = (1 - agent.discount) * h_t + agent.discount * jnp.maximum(h_t, next_vh_t)

        def Qh_loss(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qhs_s = agent.safe_critic.apply_fn({"params": safe_critic_params}, z_s)
            qhs_s_proj = agent.safe_critic.apply_fn({"params": safe_critic_params}, z_s_proj)
            qhs_t = agent.safe_critic.apply_fn({"params": safe_critic_params}, z_t)

            source_loss = (jnp.abs(qhs_s - y_h_s) + jnp.abs(qhs_s_proj - y_h_s)).mean()
            target_loss = jnp.abs(qhs_t - y_h_t).mean()
            qh_loss = source_loss + agent.lambda_t * target_loss
            return qh_loss, {
                "qh_loss": qh_loss,
                "qh_source_loss": source_loss,
                "qh_target_loss": target_loss,
                "q_h": qhs_s.mean(),
                "q_h_proj": qhs_s_proj.mean(),
                "q_h_target": qhs_t.mean(),
            }

        qh_grads, qh_info = jax.grad(Qh_loss, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=qh_grads)

        safe_target_critic_params = optax.incremental_update(safe_critic.params, agent.safe_target_critic.params, agent.tau)
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)

        new_agent = agent.replace(safe_value=safe_value, safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, {**vh_info, **qh_info}

    def update_r(agent, batch: DatasetDict, key: jnp.ndarray) -> Tuple["CBFTransfer", Dict[str, float]]:
        """Budget-conditioned CBF.update_r on the pooled D_s u D_t batch; the feasibility
        gate reads the latent-valued Q_h^phi."""
        key_v, key_q = jax.random.split(key)
        observations = batch["observations"]
        actions = batch["actions"]
        next_observations = batch["next_observations"]
        raw_cost = batch["cost_raw"][..., None]

        action_cost = agent.cost_target_critic.apply_fn(
            {"params": agent.cost_target_critic.params}, observations, actions
        ).max(axis=0)  # (B,)

        # Step 1: Update V_r  (conditioned on a random absolute budget)
        deltabar_v = sample_deltabar(key_v, (observations.shape[0], 1), action_cost[..., None], agent.budget_max)
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params}, observations, actions, deltabar_v
        )
        q = jnp.min(qs, axis=0)

        def Vr_loss(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, observations, deltabar_v)
            loss = expectile_loss(q - v, agent.reward_tau).mean()
            return loss, {"v_r_loss": loss, "v_r": v.mean()}

        vr_grads, vr_info = jax.grad(Vr_loss, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=vr_grads)

        # Step 2: Update Q_r  (budget propagated through the Bellman recursion)
        next_state_cost = agent.cost_value.apply_fn(
            {"params": agent.cost_value.params}, next_observations
        )[..., None]
        next_state_cost = jnp.maximum(next_state_cost, (action_cost[..., None] - raw_cost) / agent.cost_discount)
        next_state_cost = jnp.maximum(next_state_cost, 0.0)

        deltabar_target = sample_deltabar(key_q, (observations.shape[0], 1), next_state_cost, agent.budget_max)
        deltabar = raw_cost + deltabar_target * agent.cost_discount

        next_vr = value.apply_fn({"params": value.params}, next_observations, deltabar_target)

        z = jax.lax.stop_gradient(agent._encode(observations, actions))
        qh = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, z)
        qh_min = jnp.min(qh, axis=0)

        # budget gap B(s,a,delta) = Q_C(s,a) - delta (cf. soft_constraint_cbf.tex eq:budget_gap)
        budget_gap = action_cost - deltabar[..., 0]
        budget_covers = budget_gap <= 0
        # feasible if reachability-safe (Q_h^phi <= 0) OR the budget covers the expected cumulative cost
        feasible = jnp.logical_or(qh_min <= 0, budget_covers)

        safe_target = batch["rewards"] + agent.discount * batch["masks"] * next_vr
        # soft TD target penalizes by the budget gap itself (cf. eq:soft_td_target), not Q_h^phi
        unsafe_target = agent.r_min / (1 - agent.discount) - budget_gap
        target_q = jnp.where(feasible, safe_target, unsafe_target)

        def Qr_loss(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn({"params": critic_params}, observations, actions, deltabar)
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"q_r_loss": loss, "q_r": qs.mean(), "feasible_frac": feasible.mean()}

        qr_grads, qr_info = jax.grad(Qr_loss, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=qr_grads)

        target_critic_params = optax.incremental_update(critic.params, agent.target_critic.params, agent.tau)
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(value=value, critic=critic, target_critic=target_critic)
        return new_agent, {**vr_info, **qr_info}

    def update_actor(agent, batch: DatasetDict, key: jnp.ndarray) -> Tuple["CBFTransfer", Dict[str, float]]:
        """Budget-conditioned dual-actor update on the pooled D_s u D_t batch."""
        observations = batch["observations"]
        actions = batch["actions"]

        action_cost = agent.cost_target_critic.apply_fn(
            {"params": agent.cost_target_critic.params}, observations, actions
        ).max(axis=0)
        state_cost = agent.cost_value.apply_fn({"params": agent.cost_value.params}, observations)

        deltabar = sample_deltabar(key, (observations.shape[0], 1), action_cost[..., None], agent.budget_max)

        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params}, observations, actions, deltabar
        )
        q = qs.min(axis=0)
        v = agent.value.apply_fn({"params": agent.value.params}, observations, deltabar)

        reward_adv = q - v
        reward_weights = jnp.clip(jnp.exp(reward_adv * agent.reward_temperature), 0, 100)

        cost_adv = state_cost - action_cost
        cost_weights = jnp.clip(jnp.exp(cost_adv * agent.cost_temperature), 0, agent.cost_ub)

        deltabar_state = jnp.clip(deltabar - state_cost[..., None], 0.0, agent.budget_max)

        def reward_actor_loss_fn(actor_params: FrozenDict[str, Any]):
            dist = agent.score_model.apply_fn({"params": actor_params}, observations, deltabar_state)
            log_probs = dist.log_prob(actions)
            actor_loss = -(log_probs * reward_weights).mean()
            return actor_loss, {"reward_actor_loss": actor_loss, "reward_weights": reward_weights.mean()}

        grads, r_info = jax.grad(reward_actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        def safe_actor_loss_fn(actor_params: FrozenDict[str, Any]):
            dist = agent.safe_score_model.apply_fn({"params": actor_params}, observations)
            log_probs = dist.log_prob(actions)
            actor_loss = -(log_probs * cost_weights).mean()
            return actor_loss, {"safe_actor_loss": actor_loss, "cost_weights": cost_weights.mean()}

        grads, s_info = jax.grad(safe_actor_loss_fn, has_aux=True)(agent.safe_score_model.params)
        safe_score_model = agent.safe_score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model, safe_score_model=safe_score_model)
        return agent, {**r_info, **s_info}

    @jax.jit
    def update(self, source_batch: DatasetDict, target_batch: DatasetDict) -> Tuple["CBFTransfer", Dict[str, float]]:
        """Stage 3, run jointly each step: Q_h^phi/V_h^nu fit on source (direct +
        forward-projected) and target (direct) as in update_h; Q_C/V_C, Q_r^omega,
        V_r^psi and pi_theta then trained on a single batch pooled over D_s u D_t."""
        new_agent = self
        rng, key_r, key_a = jax.random.split(new_agent.rng, 3)
        new_agent = new_agent.replace(rng=rng)
        new_agent, h_info = new_agent.update_h(source_batch, target_batch)

        pooled_batch = {
            k: jnp.concatenate([source_batch[k], target_batch[k]], axis=0)
            for k in ("observations", "actions", "rewards", "next_observations", "masks", "cost_raw")
        }
        new_agent, c_info = new_agent.update_cost(pooled_batch)
        new_agent, r_info = new_agent.update_r(pooled_batch, key_r)
        new_agent, actor_info = new_agent.update_actor(pooled_batch, key_a)

        info = {**h_info, **c_info, **r_info, **actor_info}
        return new_agent, info

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        """Step 4: Q_h^phi is queried directly on target latents, with online budget tracking."""
        rng = self.rng
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        obs1 = jnp.expand_dims(observations, axis=0)                       # (1, obs)
        state_cost = self.cost_value.apply_fn({"params": self.cost_value.params}, obs1)  # (1,)

        deltabar_state = jnp.clip(self.delta_hat, 0.0, self.budget_max)[..., None]  # (1, 1)

        obs_N = obs1.repeat(self.N, axis=0)
        db_N = deltabar_state.repeat(self.N, axis=0)

        rng, key = jax.random.split(rng, 2)
        dist = self.score_model.apply_fn(
            {"params": self.score_model.params}, obs_N, db_N, temperature=0.1
        )
        safe_dist = self.safe_score_model.apply_fn(
            {"params": self.safe_score_model.params}, obs_N, temperature=0.1
        )
        reward_actions = dist.sample(seed=key)
        safe_actions = safe_dist.sample(seed=key)

        use_safe = (deltabar_state[0, 0] < 1e-3)
        candidates = jnp.where(use_safe, safe_actions, reward_actions)     # (N, act)

        z = self._encode(obs_N, candidates)
        qcs = compute_safe_q_latent(self.safe_target_critic.apply_fn, self.safe_target_critic.params, z)
        idx = jnp.argmin(qcs)
        action = candidates[idx]

        action_cost = self.cost_target_critic.apply_fn(
            {"params": self.cost_target_critic.params}, obs1, action[None]
        ).max(axis=0)
        action_cost = jnp.maximum(action_cost, state_cost)
        new_delta_hat = jnp.maximum(self.delta_hat + (state_cost - action_cost), 0.0) / self.cost_discount

        return action.squeeze(), self.replace(rng=rng, delta_hat=new_delta_hat)
