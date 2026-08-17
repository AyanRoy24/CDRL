from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax import struct
from flax.training.train_state import TrainState

from cbf import CBF, expectile_loss
from dataset import DatasetDict
from networks import MLP, Ensemble, StateActionValue, StateValue, get_weight_decay_mask
from networks import DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler, vp_beta_schedule
from networks.transfer_nets import AutoEncoder
from flow_transfer import _integrate_flow


@partial(jax.jit, static_argnames=("safe_critic_fn",))
def compute_safe_q_latent(safe_critic_fn, safe_critic_params, z):
    safe_q_values = safe_critic_fn({"params": safe_critic_params}, z)
    return safe_q_values.max(axis=0)


class CBFTransfer(CBF):
    """Step 3-4: Q_h^phi is defined over the shared latent Z instead of raw S x A.

    Q_r, V_r and pi_theta are left exactly as in CBF: they only ever see raw
    (s, a) from the source dataset, which is what makes them domain-invariant
    (see problem statement / Novelty). Only the safety critic and the target
    branch of the reward critic route through the frozen encoder + flow map.
    """

    autoencoder: TrainState
    velocity: TrainState
    flow_steps: int = struct.field(pytree_node=False)
    latent_dim: int = struct.field(pytree_node=False)

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
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, safe_critic_key, safe_value_key = jax.random.split(rng, 6)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        preprocess_time_cls = partial(FourierFeatures, output_size=time_dim, learnable=True)
        cond_model_cls = partial(MLP, hidden_dims=(128, 128), activations=nn.relu, activate_final=False)
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        base_model_cls = partial(MLP, hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]), activations=nn.relu, activate_final=False)
        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls, cond_encoder_cls=cond_model_cls, reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis=0)
        actor_params = FrozenDict(actor_def.init(actor_key, observations, actions, time)["params"])

        score_model = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adamw(
                learning_rate=actor_lr,
                weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                mask=get_weight_decay_mask,
            ),
        )

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=critic_layer_norm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = FrozenDict(critic_def.init(critic_key, observations, actions)["params"])
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

        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        value_params = FrozenDict(value_def.init(value_key, observations)["params"])
        value_optimiser = optax.adam(learning_rate=value_lr)
        value = TrainState.create(apply_fn=value_def.apply, params=value_params, tx=value_optimiser)

        safe_value_def = StateValue(base_cls=value_base_cls)
        safe_value_params = FrozenDict(safe_value_def.init(safe_value_key, observations)["params"])
        safe_value = TrainState.create(apply_fn=safe_value_def.apply, params=safe_value_params, tx=value_optimiser)

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
        )

    def _encode(self, observations, actions):
        return self.autoencoder.apply_fn({"params": self.autoencoder.params}, observations, actions, method=AutoEncoder.encode)

    def update_h(agent, batch: DatasetDict) -> Tuple["CBFTransfer", Dict[str, float]]:
        """Step 3(a)-(c): fit Q_h^phi to agree with the same target at z and at its forward projection f_eta(z)."""
        z = jax.lax.stop_gradient(agent._encode(batch["observations"], batch["actions"]))
        z_proj = jax.lax.stop_gradient(_integrate_flow(agent.velocity.apply_fn, agent.velocity.params, z, agent.flow_steps))

        qcs = agent.safe_target_critic.apply_fn({"params": agent.safe_target_critic.params}, z)
        qc = qcs.max(axis=0)

        def Vh_loss(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vh = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])
            loss = expectile_loss(qc - vh, agent.cost_tau).mean()
            return loss, {"vh_loss": loss, "v_h": vh.mean()}

        vh_grads, vh_info = jax.grad(Vh_loss, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=vh_grads)

        next_vh = safe_value.apply_fn({"params": safe_value.params}, batch["next_observations"])
        h_sa = batch["costs"]
        target_qh = (1 - agent.discount) * h_sa + agent.discount * jnp.maximum(h_sa, next_vh)

        def Qh_loss(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qhs = agent.safe_critic.apply_fn({"params": safe_critic_params}, z)
            qhs_proj = agent.safe_critic.apply_fn({"params": safe_critic_params}, z_proj)
            qh_loss = (jnp.abs(qhs - target_qh) + jnp.abs(qhs_proj - target_qh)).mean()
            return qh_loss, {"qh_loss": qh_loss, "q_h": qhs.mean(), "q_h_proj": qhs_proj.mean()}

        qh_grads, qh_info = jax.grad(Qh_loss, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=qh_grads)

        safe_target_critic_params = optax.incremental_update(safe_critic.params, agent.safe_target_critic.params, agent.tau)
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)

        new_agent = agent.replace(safe_value=safe_value, safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, {**vh_info, **qh_info}

    def update_r(agent, batch: DatasetDict) -> Tuple["CBFTransfer", Dict[str, float]]:
        """Same as CBF.update_r, except the Q_h^phi(z) branch condition now reads the latent-valued safety critic."""
        qs = agent.target_critic.apply_fn({"params": agent.target_critic.params}, batch["observations"], batch["actions"])
        q = jnp.min(qs, axis=0)

        def Vr_loss(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            loss = expectile_loss(q - v, agent.reward_tau).mean()
            return loss, {"v_r_loss": loss, "v_r": v.mean()}

        vr_grads, vr_info = jax.grad(Vr_loss, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=vr_grads)

        z = jax.lax.stop_gradient(agent._encode(batch["observations"], batch["actions"]))
        qh = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, z)
        qh_min = jnp.min(qh, axis=0)

        next_vr = value.apply_fn({"params": value.params}, batch["next_observations"])

        safe_mask = qh_min <= 0
        unsafe_mask = qh_min > 0
        safe_target = batch["rewards"] + agent.discount * batch["masks"] * next_vr
        unsafe_target = agent.r_min / (1 - agent.discount) - qh_min
        target_q = safe_mask * safe_target + unsafe_mask * unsafe_target

        def Qr_loss(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn({"params": critic_params}, batch["observations"], batch["actions"])
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"q_r_loss": loss, "q_r": qs.mean()}

        qr_grads, qr_info = jax.grad(Qr_loss, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=qr_grads)

        target_critic_params = optax.incremental_update(critic.params, agent.target_critic.params, agent.tau)
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(value=value, critic=critic, target_critic=target_critic)
        return new_agent, {**vr_info, **qr_info}

    def update_actor(agent, batch: DatasetDict) -> Tuple["CBFTransfer", Dict[str, float]]:
        """Same as CBF.update_actor: pi_theta trains purely on raw (s, a, r, s'), domain-invariant by construction."""
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch["actions"].shape[0],), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(key, (batch["actions"].shape[0], agent.action_dim))
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

        qs = agent.target_critic.apply_fn({"params": agent.target_critic.params}, batch["observations"], batch["actions"])
        q = qs.min(axis=0)
        v = agent.value.apply_fn({"params": agent.value.params}, batch["observations"])

        reward_adv = q - v
        reward_weights = jnp.exp(reward_adv * agent.reward_temperature)
        weights = jnp.clip(reward_weights, 0, 100)

        def actor_loss_fn(score_model_params):
            eps_pred = agent.score_model.apply_fn(
                {"params": score_model_params},
                batch["observations"],
                noisy_actions,
                time,
                rngs={"dropout": key},
                training=True,
            )
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1) * weights).mean()
            return actor_loss, {"actor_loss": actor_loss, "weights": weights.mean()}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        agent = agent.replace(score_model=score_model, rng=rng)
        return agent, info

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        """Step 4: Q_h^phi is queried directly on target latents, with no separate projection step."""
        rng = self.rng
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis=0)
        score_params = self.score_model.params
        actions, rng = ddpm_sampler(
            self.score_model.apply_fn,
            score_params,
            self.T,
            rng,
            self.action_dim,
            observations,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.M,
            self.clip_sampler,
        )
        z = self._encode(observations, actions)
        qcs = compute_safe_q_latent(self.safe_target_critic.apply_fn, self.safe_target_critic.params, z)
        idx = jnp.argmin(qcs)
        action = actions[idx]
        return action.squeeze(), self.replace(rng=rng)
