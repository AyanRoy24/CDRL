from functools import partial
from typing import Dict, Sequence, Tuple

import os
import pickle
import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from dataset import DatasetDict
from networks import MLP, StateValue
from networks.transfer_nets import AutoEncoder, VelocityField


def _integrate_flow(velocity_apply_fn, velocity_params, z0: jnp.ndarray, num_steps: int) -> jnp.ndarray:
    """f_eta(z0) = psi_eta(z0, 1), the endpoint of the ODE trajectory, via fixed-step Euler integration."""
    dt = 1.0 / num_steps

    def step(z, i):
        tau = jnp.full((z.shape[0],), i * dt)
        v = velocity_apply_fn({"params": velocity_params}, z, tau)
        return z + dt * v, None

    zT, _ = jax.lax.scan(step, z0, jnp.arange(num_steps))
    return zT


def _sinkhorn(cost: jnp.ndarray, reg: float, num_iters: int) -> jnp.ndarray:
    """Entropic-regularized solve for A* over the transportation polytope B_k with uniform marginals."""
    k0, k1 = cost.shape
    K = jnp.exp(-cost / reg)
    a = jnp.ones(k0) / k0
    b = jnp.ones(k1) / k1

    def step(carry, _):
        u, v = carry
        u = a / (K @ v + 1e-8)
        v = b / (K.T @ u + 1e-8)
        return (u, v), None

    (u, v), _ = jax.lax.scan(step, (jnp.ones(k0), jnp.ones(k1)), None, length=num_iters)
    return u[:, None] * K * v[None, :]


def _sample_coupling(key: jax.random.PRNGKey, A: jnp.ndarray, num_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Draws (i_l, j_l) ~ A* pairs, treating the coupling matrix as a joint distribution over index pairs."""
    k0, k1 = A.shape
    probs = A.reshape(-1)
    probs = probs / (probs.sum() + 1e-12)
    idx = jax.random.categorical(key, jnp.log(probs + 1e-12), shape=(num_samples,))
    return idx // k1, idx % k1


class FlowTransfer(struct.PyTreeNode):
    """Steps 1-2: a shared encoder/decoder and a safety-consistent flow map between source and target latents."""

    autoencoder: TrainState
    velocity: TrainState
    safety_regressor: TrainState
    rng: jnp.ndarray
    lambda_c: float
    lambda_h: float
    latent_dim: int = struct.field(pytree_node=False)
    ot_reg: float = struct.field(pytree_node=False)
    ot_iters: int = struct.field(pytree_node=False)
    flow_steps: int = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        latent_dim: int = 32,
        hidden_dims: Sequence[int] = (256, 256),
        autoencoder_lr: float = 1e-3,
        velocity_lr: float = 1e-3,
        regressor_lr: float = 1e-3,
        lambda_c: float = 1.0,
        lambda_h: float = 1.0,
        ot_reg: float = 0.05,
        ot_iters: int = 50,
        flow_steps: int = 10,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, ae_key, vel_key, reg_key = jax.random.split(rng, 4)

        observations = jnp.expand_dims(observation_space.sample(), axis=0)
        actions = jnp.expand_dims(action_space.sample(), axis=0)
        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        autoencoder_def = AutoEncoder(hidden_dims=hidden_dims, latent_dim=latent_dim, obs_dim=obs_dim, action_dim=action_dim)
        autoencoder_params = FrozenDict(autoencoder_def.init(ae_key, observations, actions)["params"])
        autoencoder = TrainState.create(
            apply_fn=autoencoder_def.apply, params=autoencoder_params, tx=optax.adam(learning_rate=autoencoder_lr)
        )

        velocity_def = VelocityField(hidden_dims=hidden_dims, latent_dim=latent_dim)
        z_dummy = jnp.zeros((1, latent_dim))
        tau_dummy = jnp.zeros((1,))
        velocity_params = FrozenDict(velocity_def.init(vel_key, z_dummy, tau_dummy)["params"])
        velocity = TrainState.create(apply_fn=velocity_def.apply, params=velocity_params, tx=optax.adam(learning_rate=velocity_lr))

        regressor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        regressor_def = StateValue(base_cls=regressor_base_cls)
        regressor_params = FrozenDict(regressor_def.init(reg_key, z_dummy)["params"])
        safety_regressor = TrainState.create(
            apply_fn=regressor_def.apply, params=regressor_params, tx=optax.adam(learning_rate=regressor_lr)
        )

        return cls(
            autoencoder=autoencoder,
            velocity=velocity,
            safety_regressor=safety_regressor,
            rng=rng,
            lambda_c=lambda_c,
            lambda_h=lambda_h,
            latent_dim=latent_dim,
            ot_reg=ot_reg,
            ot_iters=ot_iters,
            flow_steps=flow_steps,
        )

    def encode(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        return self.autoencoder.apply_fn({"params": self.autoencoder.params}, observations, actions, method=AutoEncoder.encode)

    def flow_map(self, z0: jnp.ndarray) -> jnp.ndarray:
        return _integrate_flow(self.velocity.apply_fn, self.velocity.params, z0, self.flow_steps)

    @jax.jit
    def update_reconstruction(self, batch: DatasetDict) -> Tuple["FlowTransfer", Dict[str, float]]:
        """Step 1: L_reconstruction over Ds u Dt, fitting phi_enc and d_dec jointly."""

        def loss_fn(params):
            _, recon = self.autoencoder.apply_fn({"params": params}, batch["observations"], batch["actions"])
            target = jnp.concatenate([batch["observations"], batch["actions"]], axis=-1)
            loss = jnp.mean((recon - target) ** 2)
            return loss, {"recon_loss": loss}

        grads, info = jax.grad(loss_fn, has_aux=True)(self.autoencoder.params)
        autoencoder = self.autoencoder.apply_gradients(grads=grads)
        return self.replace(autoencoder=autoencoder), info

    @jax.jit
    def pretrain_safety_regressor(self, target_batch: DatasetDict) -> Tuple["FlowTransfer", Dict[str, float]]:
        """Step 2(g): fit h_hat_t^xi on Dt, on top of the now-frozen encoder."""
        z = jax.lax.stop_gradient(self.encode(target_batch["observations"], target_batch["actions"]))
        h_t = target_batch["costs"]

        def loss_fn(params):
            h_pred = self.safety_regressor.apply_fn({"params": params}, z)
            loss = jnp.mean((h_pred - h_t) ** 2)
            return loss, {"reg_loss": loss}

        grads, info = jax.grad(loss_fn, has_aux=True)(self.safety_regressor.params)
        safety_regressor = self.safety_regressor.apply_gradients(grads=grads)
        return self.replace(safety_regressor=safety_regressor), info

    @jax.jit
    def update_flow(self, source_batch: DatasetDict, target_batch: DatasetDict) -> Tuple["FlowTransfer", Dict[str, float]]:
        """Step 2(a)-(h): safety-weighted OT coupling, then L_FM + lambda_h * L_align on v_eta."""
        rng, pair_key, tau_key = jax.random.split(self.rng, 3)

        z0 = jax.lax.stop_gradient(self.encode(source_batch["observations"], source_batch["actions"]))
        z1 = jax.lax.stop_gradient(self.encode(target_batch["observations"], target_batch["actions"]))
        hs = source_batch["costs"]
        ht = target_batch["costs"]

        sq_dist = jnp.sum((z0[:, None, :] - z1[None, :, :]) ** 2, axis=-1)
        h_gap = jnp.abs(hs[:, None] - ht[None, :])
        cost_matrix = sq_dist + self.lambda_c * h_gap
        A = jax.lax.stop_gradient(_sinkhorn(cost_matrix, self.ot_reg, self.ot_iters))

        k = z0.shape[0]
        i_idx, j_idx = _sample_coupling(pair_key, A, k)
        z0_pair = z0[i_idx]
        z1_pair = z1[j_idx]
        hs_pair = hs[i_idx]

        tau = jax.random.uniform(tau_key, (k,))
        z_tau = (1 - tau[:, None]) * z0_pair + tau[:, None] * z1_pair
        u = z1_pair - z0_pair

        def loss_fn(params):
            v_pred = self.velocity.apply_fn({"params": params}, z_tau, tau)
            fm_loss = jnp.mean(jnp.sum((v_pred - u) ** 2, axis=-1))

            v0 = self.velocity.apply_fn({"params": params}, z0_pair, jnp.zeros((k,)))
            z1_hat = z0_pair + v0
            h_pred = self.safety_regressor.apply_fn({"params": self.safety_regressor.params}, z1_hat)
            align_loss = jnp.mean((hs_pair - h_pred) ** 2)

            loss = fm_loss + self.lambda_h * align_loss
            return loss, {"fm_loss": fm_loss, "align_loss": align_loss}

        grads, info = jax.grad(loss_fn, has_aux=True)(self.velocity.params)
        velocity = self.velocity.apply_gradients(grads=grads)
        return self.replace(velocity=velocity, rng=rng), info

    def save(self, modeldir, save_time):
        file_name = "flow_transfer" + str(save_time) + ".pickle"
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), "wb"))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, "rb"))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent
