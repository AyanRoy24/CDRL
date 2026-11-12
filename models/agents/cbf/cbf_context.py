"""Stage 2 of the pooled pipeline with the plain hard-constraint CBF (`cbf_plain.CBF`, a copy of
CBF/SCBF/jaxrl5/agents/cbf_g.py) as the learner (instead of `IQLPooled`).

COSTA uses one fixed cost limit per map, so there is no budget machinery: the learner is the
reachability barrier Q_h/V_h + reward critic Q_r + actor, and the safety filter is "sample N
actions, keep the lowest Q_h".  The only addition is the pooled
constraint context: the frozen Stage-1 encoder q_xi infers `zeta`, which is concatenated to
the observation everywhere CBF reads it (CBF is built on an observation space widened by
`context_dim`).

Exposes `IQLPooled`'s interface (update / set_context / infer_context / reset_budget /
step_budget / eval_actions / save / load), so train_gym / evaluate_pooled run it unchanged.
"""
import os
import pickle
from typing import Sequence

import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.training.train_state import TrainState

import inspect

from models.agents.cbf.cbf_plain import CBF


def _cat(*xs):
    return jnp.concatenate(xs, axis=-1)


class CBFContext(struct.PyTreeNode):
    cbf: CBF
    context_encoder: TrainState = None
    ctx: jnp.ndarray = None
    context_dim: int = struct.field(pytree_node=False, default=8)
    seg_m: int = struct.field(pytree_node=False, default=16)

    @classmethod
    def create(cls, seed, observation_space, action_space, context_encoder, context_dim,
               seg_m=16, **cbf_kwargs):
        obs_dim = observation_space.shape[0]
        wide = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim + context_dim,), dtype=np.float32)
        accepted = inspect.signature(CBF.create).parameters   # drop the budget/cost-critic kwargs of cbf_g
        cbf = CBF.create(seed, wide, action_space, **{k: v for k, v in cbf_kwargs.items() if k in accepted})
        return cls(cbf=cbf, context_encoder=context_encoder, ctx=jnp.zeros((context_dim,)),
                   context_dim=context_dim, seg_m=seg_m)

    # ----------------------------------------------------------------- helpers
    def infer_context(self, segments):
        z = self.context_encoder.apply_fn({"params": self.context_encoder.params}, segments)
        return jax.lax.stop_gradient(z)

    def set_context(self, ctx_vec):
        return self.replace(ctx=jnp.asarray(ctx_vec, dtype=jnp.float32))

    @property
    def delta_hat(self):
        return 0.0   # no running budget (evaluate_pooled only logs it)

    # ------------------------------------------------------------------ update
    @jax.jit
    def update(self, source_batch, target_batch, ctx_s, ctx_t):
        """One CBF step on the pooled batch, each transition conditioned on its own
        regime's zeta.  `target_batch = ctx_t = None` is the general policy (source only)."""
        keys = ("observations", "actions", "rewards", "next_observations", "masks", "costs", "cost_raw")
        if ctx_s.ndim == 1:
            ctx_s = jnp.broadcast_to(ctx_s, (source_batch["observations"].shape[0], self.context_dim))
        if target_batch is None:
            pooled = {k: source_batch[k] for k in keys}
            ctx = ctx_s
        else:
            if ctx_t.ndim == 1:
                ctx_t = jnp.broadcast_to(ctx_t, (target_batch["observations"].shape[0], self.context_dim))
            pooled = {k: jnp.concatenate([source_batch[k], target_batch[k]], axis=0) for k in keys}
            ctx = jnp.concatenate([ctx_s, ctx_t], axis=0)
        pooled["observations"] = _cat(pooled["observations"], ctx)
        pooled["next_observations"] = _cat(pooled["next_observations"], ctx)
        cbf, info = self.cbf.update(pooled)
        return self.replace(cbf=cbf), info

    # -------------------------------------------------------------- deployment
    def reset_budget(self, obs0, delta_init):
        """No-op: the hard-constraint CBF has no budget; the threshold only normalises scores."""
        return self

    def step_budget(self, obs, action, next_obs):
        """No-op (no running budget)."""
        return self

    @jax.jit
    def eval_actions(self, observations):
        obs_ctx = _cat(observations, self.ctx)
        action, cbf = self.cbf.eval_actions(obs_ctx)
        return action, self.replace(cbf=cbf)

    def save(self, modeldir, save_time):
        file_name = "model" + str(save_time) + ".pickle"
        pickle.dump(flax.serialization.to_state_dict(self), open(os.path.join(modeldir, file_name), "wb"))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, "rb"))
        return flax.serialization.from_state_dict(target=self, state=pkl_file)
