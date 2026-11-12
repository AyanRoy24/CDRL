"""Stage 1, CoPDT style: train the constraint-context encoder (CDSRL/CoPDT/OSRL
`ContextEncoderTrainer`, train_context_encoder.py) -- replacing the COSTA-style
`ContextEncoderTrainer` (distance-metric loss on regime labels + counterfactual relabeling by
per-goal cost *discriminators*).

There is no discriminator, no regime label and no relabeling here.  A context is a set of
`context_size` transitions (s, a, s', r) of one task, each tagged by its cost; the encoder maps
it to zeta (`SafetyAwareEncoder`) and a `MultiHeadDecoder` must predict, from (zeta, s, a) for
every transition of the set, the next-state latent, the reward and whether the step was
unsafe:

    loss = w_s * MSE(s'_pred, s') + w_r * MSE(r_pred, r) + w_c * BCE(cost_pred, cost > 0)

so zeta has to carry the task's dynamics / reward / constraint.  Everything works in the shared
latent space of the frozen state / action auto-encoders (state_encoding_dim=32, action 2).
After training only the encoder is kept (`load_context_encoder`).
"""

import json
import os
import pickle

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from models.networks.context_encoder import MultiHeadDecoder, SafetyAwareEncoder, SimpleMlpEncoder


def context_input_dim(se, ae, simple_mlp=False):
    """Per-transition encoder input is (s, a, s', r) -- the cost is passed separately (it only
    routes rows between the safe / unsafe branch), or appended for the simple-MLP variant."""
    return 2 * se + ae + 1


def build_encoder_def(cfg):
    if cfg.get("simple_mlp", False):
        return SimpleMlpEncoder(hidden_dims=tuple(cfg["context_encoder_hidden_sizes"]),
                                output_dim=cfg["context_encoding_dim"])
    return SafetyAwareEncoder(hidden_dims=tuple(cfg["context_encoder_hidden_sizes"]),
                              output_dim=cfg["context_encoding_dim"], simple_gate=cfg.get("simple_gate", False))


class CoPDTContextTrainer(struct.PyTreeNode):
    encoder: TrainState
    decoder: TrainState
    se: int = struct.field(pytree_node=False)
    ae: int = struct.field(pytree_node=False)
    state_loss_weight: float = struct.field(pytree_node=False, default=1.0)
    reward_loss_weight: float = struct.field(pytree_node=False, default=0.0)
    cost_loss_weight: float = struct.field(pytree_node=False, default=2.0)

    @classmethod
    def create(cls, seed, cfg):
        """cfg: state_encoding_dim, action_encoding_dim, context_encoding_dim,
        context_encoder_hidden_sizes, feature_hidden_sizes, decoder_hidden_sizes, simple_gate,
        simple_mlp, loss weights, learning_rate / min_learning_rate / decay_step / decay_rate / betas."""
        se, ae, cd = cfg["state_encoding_dim"], cfg["action_encoding_dim"], cfg["context_encoding_dim"]
        k_enc, k_dec = jax.random.split(jax.random.PRNGKey(seed))
        enc_def = build_encoder_def(cfg)
        d_in = context_input_dim(se, ae)
        enc_params = enc_def.init(k_enc, jnp.zeros((1, 4, d_in)), jnp.zeros((1, 4)))["params"]
        dec_def = MultiHeadDecoder(feature_hidden_dims=tuple(cfg["feature_hidden_sizes"]),
                                   decoder_hidden_dims=tuple(cfg["decoder_hidden_sizes"]), state_dim=se)
        dec_params = dec_def.init(k_dec, jnp.zeros((1, 4, cd + se + ae)))["params"]

        # torch StepLR(step_size=decay_step, gamma=decay_rate), frozen once it reaches min_learning_rate
        lr, lo, step, rate = cfg["learning_rate"], cfg["min_learning_rate"], cfg["decay_step"], cfg["decay_rate"]
        schedule = lambda t: jnp.maximum(lr * rate ** (t // step), lo)
        b1, b2 = cfg["betas"]
        tx = lambda: optax.adamw(schedule, b1=b1, b2=b2, eps=1e-5, weight_decay=0.01)
        return cls(
            encoder=TrainState.create(apply_fn=enc_def.apply, params=enc_params, tx=tx()),
            decoder=TrainState.create(apply_fn=dec_def.apply, params=dec_params, tx=tx()),
            se=se, ae=ae,
            state_loss_weight=cfg["state_loss_weight"], reward_loss_weight=cfg["reward_loss_weight"],
            cost_loss_weight=cfg["cost_loss_weight"],
        )

    def embed(self, x, cost):
        """x: (M, N, 2se+ae+1), cost: (M, N) -> zeta (M, context_dim)."""
        return self.encoder.apply_fn({"params": self.encoder.params}, x, cost)

    @jax.jit
    def update(self, x, cost):
        se, ae = self.se, self.ae
        s, a = x[..., :se], x[..., se:se + ae]
        ns, r = x[..., se + ae:2 * se + ae], x[..., -1]
        unsafe = (cost > 0).astype(jnp.float32)

        def loss_fn(enc_params, dec_params):
            zeta = self.encoder.apply_fn({"params": enc_params}, x, cost)             # (M, ctx)
            zeta = jnp.broadcast_to(zeta[:, None, :], (x.shape[0], x.shape[1], zeta.shape[-1]))
            ns_pred, r_pred, c_prob = self.decoder.apply_fn(
                {"params": dec_params}, jnp.concatenate([zeta, s, a], axis=-1))
            state_loss = jnp.mean((ns_pred - jax.lax.stop_gradient(ns)) ** 2)
            reward_loss = jnp.mean((r_pred - jax.lax.stop_gradient(r)) ** 2)
            c_prob = jnp.clip(c_prob, 1e-6, 1 - 1e-6)
            cost_loss = -jnp.mean(unsafe * jnp.log(c_prob) + (1 - unsafe) * jnp.log(1 - c_prob))
            loss = (self.state_loss_weight * state_loss + self.reward_loss_weight * reward_loss
                    + self.cost_loss_weight * cost_loss)
            acc = jnp.mean(((c_prob > 0.5).astype(jnp.float32) == unsafe).astype(jnp.float32))
            return loss, {"all_loss": loss, "next_state_loss": state_loss, "reward_loss": reward_loss,
                          "cost_loss": cost_loss, "cost_acc": acc}

        (enc_grads, dec_grads), info = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)(
            self.encoder.params, self.decoder.params)
        new = self.replace(encoder=self.encoder.apply_gradients(grads=enc_grads),
                           decoder=self.decoder.apply_gradients(grads=dec_grads))
        return new, info

    def save(self, outdir, cfg, extra=None):
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "context_encoder.pickle"), "wb") as f:
            pickle.dump({"encoder": jax.device_get(self.encoder.params),
                         "decoder": jax.device_get(self.decoder.params)}, f)
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump({**cfg, **(extra or {})}, f, indent=4)


def load_context_encoder(ctx_dir):
    """(embed, cfg): `embed(x, cost)` is the jitted frozen encoder (x (M, N, 2se+ae+1),
    cost (M, N)) -> zeta (M, context_dim); `cfg` the saved training config."""
    with open(os.path.join(ctx_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(ctx_dir, "context_encoder.pickle"), "rb") as f:
        params = pickle.load(f)["encoder"]
    enc_def = build_encoder_def(cfg)
    embed = jax.jit(lambda x, cost: enc_def.apply({"params": params}, x, cost))
    return embed, cfg
