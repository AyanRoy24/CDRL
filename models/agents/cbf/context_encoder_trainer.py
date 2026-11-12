"""Stage 1 of Sec. `sec:pooled-algo`: train the constraint-context encoder q_xi,
matching CDSRL/COSTA/COSTA's `OfflineMetaLearner` (FOCAL-style, `is_focal=True`,
`use_cost_dynamic=False`) exactly in objective, not just architecture:

  * a **distance-metric loss** (DML) on raw (non-normalised) embeddings: segments
    drawn from the same constraint regime are pulled together
    (sqrt(mean((z_i-z_j)^2) + eps)), segments from different regimes are pushed
    apart (1 / (sqrt(mean((z_i-z_j)^2) + eps/1e4) + eps*100)) -- COSTA's
    `distance_metric_loss`, not a softmax-temperature InfoNCE/SupCon loss;

  * with COSTA's cost-dynamics term (`use_cost_dynamic=True`, which run_context_encoding.py
    turns on by passing the per-task cost discriminators; `update_cf` here): the same loss
    on counterfactually relabeled segments -- regime k's states with ANOTHER regime's cost
    bits must embed away from regime k, and another regime's states carrying regime k's cost
    bits must embed with it.  Without it the encoder can separate regimes by their
    state/action statistics and never read the cost column (it then maps on-policy
    segments, which lie off those statistics, to one code whatever the goal).

  * an auxiliary **decoder reconstruction** term (COSTA's `Decoder`): predict the
    per-step cost bit from concat(s, a, s', zeta) with a BCE loss, jointly
    optimised with the encoder (one gradient step updates both -- COSTA runs a
    single `AdamW([{encoder params}, {decoder params}])`).

The DML term in `update` is COSTA's `use_cost_dynamic=False` path only.

After training, q_xi is frozen and only ever read (Sec. `sec:pooled-practical`,
Stage 3: "Detach the gradient from q_xi during critic fitting"); the decoder is
discarded (it exists purely to regularise q_xi during Stage 1).
"""

import os
import pickle
from typing import Sequence

import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from models.networks.context_encoder import ContextEncoder, ContextDecoder


class ContextEncoderTrainer(struct.PyTreeNode):
    encoder: TrainState
    decoder: TrainState
    rng: jnp.ndarray
    obs_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    context_dim: int = struct.field(pytree_node=False)
    m: int = struct.field(pytree_node=False)
    decoder_weight: float = struct.field(pytree_node=False, default=0.5)
    eps: float = struct.field(pytree_node=False, default=1e-3)
    contrastive_weight: float = struct.field(pytree_node=False, default=0.1)   # COSTA's weight of the counterfactual term

    @classmethod
    def create(
        cls,
        seed,
        obs_dim,
        action_dim,
        m,
        context_dim: int = 8,
        hidden_dims: Sequence[int] = (256, 256),
        decoder_hidden_dims: Sequence[int] = (256, 256),
        lr: float = 1e-3,
        decoder_weight: float = 0.5,
        contrastive_weight: float = 0.1,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, enc_key, dec_key = jax.random.split(rng, 3)
        seg_feat_dim = 2 * obs_dim + action_dim + 1  # concat(s, a, c_raw, s')

        enc_def = ContextEncoder(hidden_dims=tuple(hidden_dims), context_dim=context_dim)
        dummy_seg = jnp.zeros((1, m, seg_feat_dim))
        enc_params = FrozenDict(enc_def.init(enc_key, dummy_seg)["params"])
        encoder = TrainState.create(apply_fn=enc_def.apply, params=enc_params, tx=optax.adamw(lr, eps=1e-5))

        dec_def = ContextDecoder(hidden_dims=tuple(decoder_hidden_dims))
        dummy_dec_in = jnp.zeros((1, m, 2 * obs_dim + action_dim + context_dim))
        dec_params = FrozenDict(dec_def.init(dec_key, dummy_dec_in)["params"])
        decoder = TrainState.create(apply_fn=dec_def.apply, params=dec_params, tx=optax.adamw(lr, eps=1e-5))

        return cls(
            encoder=encoder, decoder=decoder, rng=rng,
            obs_dim=obs_dim, action_dim=action_dim, context_dim=context_dim, m=m,
            decoder_weight=decoder_weight, contrastive_weight=contrastive_weight,
        )

    def embed(self, segments):
        return self.encoder.apply_fn({"params": self.encoder.params}, segments)

    def _split(self, segments):
        o, a = self.obs_dim, self.action_dim
        obs = segments[..., :o]
        act = segments[..., o:o + a]
        craw = segments[..., o + a]
        nobs = segments[..., o + a + 1:]
        return obs, act, craw, nobs

    @jax.jit
    def update(self, segments, labels):
        """`segments`: (B, m, seg_feat_dim) = concat(s, a, c_raw, s'); `labels`:
        (B,) the regime id of each segment (dataset identity, only used to build
        positive/negative pairs here -- never fed to the critics/policy)."""
        obs, act, craw, nobs = self._split(segments)
        B = segments.shape[0]
        eye = jnp.eye(B, dtype=bool)
        same = (labels[:, None] == labels[None, :]) & (~eye)
        diff = labels[:, None] != labels[None, :]

        def loss_fn(enc_params, dec_params):
            z = self.encoder.apply_fn({"params": enc_params}, segments)  # (B, context_dim)

            sqdist = jnp.mean((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)  # (B, B)
            pos_term = jnp.sqrt(sqdist + self.eps)
            neg_term = 1.0 / (jnp.sqrt(sqdist + self.eps / 1e4) + self.eps * 100)
            pos_cnt = jnp.maximum(same.sum(), 1)
            neg_cnt = jnp.maximum(diff.sum(), 1)
            pos_loss = jnp.sum(jnp.where(same, pos_term, 0.0)) / pos_cnt
            neg_loss = jnp.sum(jnp.where(diff, neg_term, 0.0)) / neg_cnt
            dml_loss = pos_loss + neg_loss

            z_rep = jnp.broadcast_to(z[:, None, :], (B, self.m, z.shape[-1]))
            dec_in = jnp.concatenate([obs, act, nobs, z_rep], axis=-1)
            pred = self.decoder.apply_fn({"params": dec_params}, dec_in)  # (B, m), in (0, 1)
            pred = jnp.clip(pred, 1e-6, 1 - 1e-6)
            decoder_loss = -(craw * jnp.log(pred) + (1 - craw) * jnp.log(1 - pred)).mean()

            loss = dml_loss + self.decoder_weight * decoder_loss

            neg_sqdist = -sqdist - 1e9 * eye.astype(sqdist.dtype)
            nn_idx = jnp.argmax(neg_sqdist, axis=1)
            nn_acc = (labels[nn_idx] == labels).mean()

            return loss, {
                "ctx_loss": loss, "ctx_dml_loss": dml_loss, "ctx_pos_loss": pos_loss,
                "ctx_neg_loss": neg_loss, "ctx_decoder_loss": decoder_loss, "ctx_nn_acc": nn_acc,
            }

        (enc_grads, dec_grads), info = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)(
            self.encoder.params, self.decoder.params
        )
        encoder = self.encoder.apply_gradients(grads=enc_grads)
        decoder = self.decoder.apply_gradients(grads=dec_grads)
        return self.replace(encoder=encoder, decoder=decoder), info

    @jax.jit
    def update_cf(self, segments, labels, pos_segments, neg_segments):
        """`update` plus COSTA's counterfactual (`use_cost_dynamic`) term.  `pos_segments` /
        `neg_segments` are row-aligned with `segments` / `labels`: pos = regime labels[i]'s cost
        bits on another regime's states, neg = regime labels[i]'s own states with another
        regime's cost bits (PooledSource.sample_contrastive(counterfactual=True)).

          cpos  original_i ~ pos_j and pos_i ~ pos_j, for i, j of one regime
          cneg  original_i !~ neg_j for i, j of one regime, and pos_i !~ pos_j across regimes
          loss = DML + contrastive_weight * (cpos + cneg) + decoder_weight * BCE
        with the decoder, as in COSTA, on the original segments only."""
        obs, act, craw, nobs = self._split(segments)
        B = segments.shape[0]
        eye = jnp.eye(B, dtype=bool)
        same_all = labels[:, None] == labels[None, :]
        same = same_all & (~eye)
        diff = ~same_all
        eps = self.eps

        def sq(a, b):
            return jnp.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)

        def mean_where(mask, x):
            return jnp.sum(jnp.where(mask, x, 0.0)) / jnp.maximum(mask.sum(), 1)

        def loss_fn(enc_params, dec_params):
            enc = lambda x: self.encoder.apply_fn({"params": enc_params}, x)
            z, zp, zn = enc(segments), enc(pos_segments), enc(neg_segments)

            d_zz = sq(z, z)
            pos_loss = mean_where(same, jnp.sqrt(d_zz + eps))
            neg_loss = mean_where(diff, 1.0 / (jnp.sqrt(d_zz + eps / 1e4) + eps * 100))
            dml_loss = pos_loss + neg_loss

            d_zp, d_zn, d_pp = sq(z, zp), sq(z, zn), sq(zp, zp)
            cpos_terms = jnp.concatenate([jnp.where(same_all, jnp.sqrt(d_zp + eps), 0.0).ravel(),
                                          jnp.where(same, jnp.sqrt(d_pp + eps), 0.0).ravel()])
            cpos_cnt = same_all.sum() + same.sum()
            cneg_terms = jnp.concatenate([
                jnp.where(same_all, 1.0 / (jnp.sqrt(d_zn + eps / 1e4) + eps * 100), 0.0).ravel(),
                jnp.where(diff, 1.0 / jnp.sqrt(d_pp + eps / 100), 0.0).ravel()])
            cneg_cnt = same_all.sum() + diff.sum()
            con_loss = cpos_terms.sum() / jnp.maximum(cpos_cnt, 1) + cneg_terms.sum() / jnp.maximum(cneg_cnt, 1)

            z_rep = jnp.broadcast_to(z[:, None, :], (B, self.m, z.shape[-1]))
            dec_in = jnp.concatenate([obs, act, nobs, z_rep], axis=-1)
            pred = jnp.clip(self.decoder.apply_fn({"params": dec_params}, dec_in), 1e-6, 1 - 1e-6)
            decoder_loss = -(craw * jnp.log(pred) + (1 - craw) * jnp.log(1 - pred)).mean()

            loss = dml_loss + self.contrastive_weight * con_loss + self.decoder_weight * decoder_loss

            nn_idx = jnp.argmax(-d_zz - 1e9 * eye.astype(d_zz.dtype), axis=1)
            return loss, {
                "ctx_loss": loss, "ctx_dml_loss": dml_loss, "ctx_pos_loss": pos_loss,
                "ctx_neg_loss": neg_loss, "ctx_decoder_loss": decoder_loss,
                "ctx_nn_acc": (labels[nn_idx] == labels).mean(),
                "ctx_cf_loss": con_loss,
                # how far the relabeled copies sit from their own regime's originals: the neg one
                # (same states, other cost bits) must not collapse to 0 or the cost column is ignored
                "ctx_cf_pos_dist": mean_where(same_all, jnp.sqrt(d_zp)),
                "ctx_cf_neg_dist": mean_where(same_all, jnp.sqrt(d_zn)),
            }

        (enc_grads, dec_grads), info = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)(
            self.encoder.params, self.decoder.params
        )
        encoder = self.encoder.apply_gradients(grads=enc_grads)
        decoder = self.decoder.apply_gradients(grads=dec_grads)
        return self.replace(encoder=encoder, decoder=decoder), info

    def save(self, modeldir, save_time):
        fn = "context_encoder" + str(save_time) + ".pickle"
        pickle.dump(flax.serialization.to_state_dict(self), open(os.path.join(modeldir, fn), "wb"))

    def load(self, model_location):
        pkl = pickle.load(open(model_location, "rb"))
        return flax.serialization.from_state_dict(target=self, state=pkl)
