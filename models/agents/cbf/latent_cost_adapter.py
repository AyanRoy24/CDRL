"""The JAX half of adapt_flow.py: the general policy from train_general.py, frozen,
with its cost estimates swapped for a target cost critic Q_t(z) on the shared latent
z = phi_enc(s, a).  Reward actor, safe actor and q_xi are unchanged; only the
budget/safe-actor switch and the budget recursion read Q_t.

Q_t is fit on two terms:
  * transported source: Q_t(f_eta(z_s)) -> the general Q_C(s, zeta_s, a), where f_eta
    is the flow learned in fm_flow.py and z_s, zeta_s come from a source regime.
  * real target: TD on D_t, y = c + gamma * Q_target(phi_enc(s', pi_safe(s', zeta_t))).

The latent is over (s, a), so V_C(s) has no latent of its own.  It is read as
Q_t(phi_enc(s, pi_safe(s, zeta))): pi_safe is what the general V_C's low expectile is
extracted by, so this is the same "best-case cost from s" that IQLPooled gates on.

Exposes IQLPooled's deployment interface, so evaluate_pooled runs it unchanged.
"""

import os
import pickle
from functools import partial
from typing import Sequence

import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from models.agents.cbf.iql_pooled import IQLPooled, _cat
from models.networks import Ensemble, MLP, StateValue
from models.networks.transfer_nets import AutoEncoder


class LatentCostAdapter(struct.PyTreeNode):
    rng: jnp.ndarray
    base: IQLPooled                       # frozen general policy
    autoencoder: TrainState               # frozen phi_enc
    latent_critic: TrainState             # Q_t(z)
    latent_target_critic: TrainState
    tau: float = 0.005
    lambda_t: float = 1.0
    cost_discount: float = 0.99

    # --- deployment runtime state ----------------------------------------------
    ctx: jnp.ndarray = None
    delta_hat: float = 0.0

    @classmethod
    def create(cls, seed: int, base: IQLPooled, autoencoder: TrainState, latent_dim: int,
               hidden_dims: Sequence[int] = (256, 256), critic_lr: float = 3e-4,
               tau: float = 0.005, lambda_t: float = 1.0):
        rng, key = jax.random.split(jax.random.PRNGKey(seed))
        critic_base = partial(MLP, hidden_dims=hidden_dims, activate_final=True, use_layer_norm=True)
        critic_def = Ensemble(partial(StateValue, base_cls=critic_base), num=2)
        params = FrozenDict(critic_def.init(key, jnp.zeros((1, latent_dim)))["params"])
        critic = TrainState.create(apply_fn=critic_def.apply, params=params, tx=optax.adam(critic_lr))
        target_critic = TrainState.create(
            apply_fn=critic_def.apply, params=params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )
        return cls(
            rng=rng, base=base, autoencoder=autoencoder,
            latent_critic=critic, latent_target_critic=target_critic,
            tau=tau, lambda_t=lambda_t, cost_discount=base.cost_discount,
            ctx=jnp.zeros((base.context_dim,)), delta_hat=0.0,
        )

    # ----------------------------------------------------------------- helpers
    def _encode(self, obs, actions):
        return self.autoencoder.apply_fn({"params": self.autoencoder.params}, obs, actions, method=AutoEncoder.encode)

    def _safe_actions(self, obs, ctx):
        _, dist_safe = self.base.actor.apply_fn(
            {"params": self.base.actor.params}, _cat(obs, ctx), jnp.zeros((obs.shape[0], 1)))
        return dist_safe.mode()

    def _qt(self, obs, actions):
        qs = self.latent_target_critic.apply_fn({"params": self.latent_target_critic.params}, self._encode(obs, actions))
        return jnp.max(qs, axis=0)

    def _vt(self, obs, ctx):
        return self._qt(obs, self._safe_actions(obs, ctx))

    # ------------------------------------------------------------------ update
    @jax.jit
    def update(self, z_proj, y_proj, target_batch, ctx_t):
        """One Q_t step on a transported source batch (f_eta(z_s), teacher Q_C) and a D_t batch."""
        ctx = jnp.broadcast_to(ctx_t, (target_batch["observations"].shape[0], self.base.context_dim))
        z_t = self._encode(target_batch["observations"], target_batch["actions"])
        next_v = self._vt(target_batch["next_observations"], ctx)
        y_t = target_batch["cost_raw"] + self.cost_discount * target_batch["masks"] * next_v

        def loss_fn(params):
            q_proj = self.latent_critic.apply_fn({"params": params}, z_proj)
            q_t = self.latent_critic.apply_fn({"params": params}, z_t)
            proj_loss = ((q_proj - y_proj) ** 2).mean()
            td_loss = ((q_t - y_t) ** 2).mean()
            return proj_loss + self.lambda_t * td_loss, {
                "proj_loss": proj_loss, "td_loss": td_loss, "mean_q_target": q_t.mean(), "mean_q_proj": q_proj.mean(),
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(self.latent_critic.params)
        critic = self.latent_critic.apply_gradients(grads=grads)
        target_params = optax.incremental_update(critic.params, self.latent_target_critic.params, self.tau)
        target_critic = self.latent_target_critic.replace(params=target_params)
        return self.replace(latent_critic=critic, latent_target_critic=target_critic), info

    # -------------------------------------------------------------- deployment
    def infer_context(self, segments):
        return self.base.infer_context(segments)

    def set_context(self, ctx_vec):
        return self.replace(ctx=jnp.asarray(ctx_vec, dtype=jnp.float32))

    def reset_budget(self, obs0, delta_init):
        return self.replace(delta_hat=float(delta_init))

    @jax.jit
    def step_budget(self, obs, action, next_obs):
        """g(s,a,s',delta) = V_t(s') + (delta - Q_t(s,a)) / gamma."""
        ctx = jnp.expand_dims(self.ctx, 0)
        qc = self._qt(jnp.expand_dims(obs, 0), jnp.expand_dims(action, 0))[0]
        next_vc = self._vt(jnp.expand_dims(next_obs, 0), ctx)[0]
        return self.replace(delta_hat=next_vc + (self.delta_hat - qc) / self.cost_discount)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        """IQLPooled.eval_actions with V_C replaced by V_t: the budget-conditioned actor
        when the remaining budget covers the target's best-case cost, else pi_safe."""
        rng, key = jax.random.split(self.rng, 2)
        obs = jnp.expand_dims(jax.device_put(observations), axis=0)
        ctx = jnp.expand_dims(self.ctx, 0)

        deltabar = jnp.clip(jnp.asarray(self.delta_hat, dtype=jnp.float32), 0.0)
        delta_in = jnp.reshape(deltabar, (1, 1))
        dist, dist_safe = self.base.actor.apply_fn(
            {"params": self.base.actor.params}, _cat(obs, ctx), delta_in, temperature=0.0)
        r_action = dist.sample(seed=key)
        safe_action = dist_safe.sample(seed=key)
        state_cost = self._qt(obs, safe_action)
        action = jnp.where(deltabar > state_cost, r_action, safe_action)
        return action.squeeze(), self.replace(rng=rng)

    def save(self, modeldir, save_time):
        file_name = "adapter" + str(save_time) + ".pickle"
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), "wb"))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, "rb"))
        return flax.serialization.from_state_dict(target=self, state=pkl_file)
