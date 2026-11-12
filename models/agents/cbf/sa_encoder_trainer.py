"""JAX port of CDSRL/CoPDT/OSRL's `ActionAETrainer` / `StateAETrainer` (osrl/algorithms/sa_encoder.py)
plus the frozen `Codec` the rest of the pipeline uses to move one task between its raw
(state, action) space and the shared latent space.

  * action AE : reconstruct a ~ N-perturbed-latent decode (add_noise), MSE.
  * state AE  : reconstruct s and s', and an inverse-dynamics head must recover the (latent)
                action from (z_s, z_s'), loss = rec(s) + rec(s') + idm_w * MSE(idm, a_latent).
Both are trained per env *family* on the concatenated data of that family's variants
(train_sa_encoder.py), mirroring train_sa_encoder_share.py.
"""

import json
import os
import pickle
from functools import partial
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from models.networks.sa_encoder import ActionAE, InverseDynamics, StateAE


# ------------------------------------------------------------------ action AE
def create_action_ae(seed, action_dim, encode_dim, hidden_sizes, lr):
    ae = ActionAE(action_dim=action_dim, encode_dim=encode_dim, hidden_sizes=tuple(hidden_sizes))
    params = ae.init(jax.random.PRNGKey(seed), jnp.zeros((1, action_dim)))["params"]
    return ae, TrainState.create(apply_fn=None, params=params, tx=optax.adam(lr))


@partial(jax.jit, static_argnames=("ae", "noise_scale"))
def action_ae_step(ts, key, ae, noise_scale, actions):
    noise = jax.random.normal(key, (actions.shape[0], ae.encode_dim)) * noise_scale

    def loss_fn(params):
        _, rec = ae.apply({"params": params}, actions, noise)
        return jnp.mean((rec - actions) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(ts.params)
    return ts.apply_gradients(grads=grads), loss


@partial(jax.jit, static_argnames=("ae",))
def action_ae_eval(params, ae, actions):
    _, rec = ae.apply({"params": params}, actions)
    return jnp.mean((rec - actions) ** 2)


# ------------------------------------------------------------------ state AE
def create_state_ae(seed, state_dim, action_latent_dim, encode_dim, hidden_sizes, idm_hidden_sizes, lr):
    ae = StateAE(state_dim=state_dim, encode_dim=encode_dim, hidden_sizes=tuple(hidden_sizes))
    idm = InverseDynamics(action_dim=action_latent_dim, hidden_sizes=tuple(idm_hidden_sizes))
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    z = jnp.zeros((1, encode_dim))
    params = {"ae": ae.init(k1, jnp.zeros((1, state_dim)))["params"],
              "idm": idm.init(k2, z, z)["params"]}
    return ae, idm, TrainState.create(apply_fn=None, params=params, tx=optax.adam(lr))


def _state_ae_loss(params, ae, idm, idm_w, s, a_lat, ns):
    z, s_rec = ae.apply({"params": params["ae"]}, s)
    nz, ns_rec = ae.apply({"params": params["ae"]}, ns)
    a_rec = idm.apply({"params": params["idm"]}, z, nz)
    rec = jnp.mean((s_rec - s) ** 2) + jnp.mean((ns_rec - ns) ** 2)
    idm_loss = jnp.mean((a_rec - a_lat) ** 2)
    return rec + idm_w * idm_loss, (rec, idm_loss)


@partial(jax.jit, static_argnames=("ae", "idm", "idm_w"))
def state_ae_step(ts, ae, idm, idm_w, s, a_lat, ns):
    (loss, (rec, idm_loss)), grads = jax.value_and_grad(_state_ae_loss, has_aux=True)(
        ts.params, ae, idm, idm_w, s, a_lat, ns)
    return ts.apply_gradients(grads=grads), {"total_loss": loss, "state_recon_loss": rec, "idm_loss": idm_loss}


@partial(jax.jit, static_argnames=("ae", "idm", "idm_w"))
def state_ae_eval(params, ae, idm, idm_w, s, a_lat, ns):
    loss, (rec, idm_loss) = _state_ae_loss(params, ae, idm, idm_w, s, a_lat, ns)
    return {"total_loss": loss, "state_recon_loss": rec, "idm_loss": idm_loss}


# ------------------------------------------------------------------- Codec
def save_sa_encoder(outdir, family, cfg, state_params, action_params=None):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "state_ae.pickle"), "wb") as f:
        pickle.dump(jax.device_get(state_params), f)
    if action_params is not None:
        with open(os.path.join(outdir, "action_ae.pickle"), "wb") as f:
            pickle.dump(jax.device_get(action_params), f)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump({**cfg, "family": family, "has_action_ae": action_params is not None}, f, indent=4)


class Codec:
    """Frozen raw <-> latent maps of one env family.

        encode_state : (.., state_dim)  -> (.., state_encode_dim)
        encode_action: (.., action_dim) -> (.., action_encode_dim)   (identity clip if no action AE)
        decode_action: (.., action_encode_dim) -> (.., action_dim)

    All three are jitted with the (frozen) parameters baked in."""

    def __init__(self, state_dim, action_dim, cfg, state_params, action_params=None):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.state_encode_dim = cfg["state_encode_dim"]
        self.action_encode_dim = cfg["action_encode_dim"] if action_params is not None else action_dim
        self.has_action_ae = action_params is not None
        sae = StateAE(state_dim=state_dim, encode_dim=cfg["state_encode_dim"],
                      hidden_sizes=tuple(cfg["state_encoder_hidden_sizes"]))
        self.encode_state = jax.jit(lambda x: sae.apply({"params": state_params}, x, method=StateAE.encode))
        if self.has_action_ae:
            aae = ActionAE(action_dim=action_dim, encode_dim=cfg["action_encode_dim"],
                           hidden_sizes=tuple(cfg["action_encoder_hidden_sizes"]))
            self.encode_action = jax.jit(lambda a: aae.apply({"params": action_params}, a, method=ActionAE.encode))
            self.decode_action = jax.jit(lambda z: aae.apply({"params": action_params}, z, method=ActionAE.decode))
        else:
            self.encode_action = jax.jit(lambda a: jnp.clip(a, -1.0, 1.0))
            self.decode_action = jax.jit(lambda z: jnp.clip(z, -1.0, 1.0))


def load_codec(sa_dir, family, state_dim, action_dim):
    d = os.path.join(sa_dir, family)
    assert os.path.isfile(os.path.join(d, "config.json")), \
        f"no state/action encoder for {family} in {d}: run train_sa_encoder.py first"
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(d, "state_ae.pickle"), "rb") as f:
        sp = pickle.load(f)
    ap = None
    if cfg.get("has_action_ae"):
        with open(os.path.join(d, "action_ae.pickle"), "rb") as f:
            ap = pickle.load(f)
    return Codec(state_dim, action_dim, cfg, sp, ap), cfg
