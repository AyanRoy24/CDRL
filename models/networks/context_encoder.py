"""q_xi : sigma -> zeta, the constraint-context encoder of Sec. `sec:pooled`.

Ported to match CDSRL/COSTA/COSTA's default context encoder exactly
(`nets/encoder.py::MLPAttnEncoder`, `encoder_type="mlp_attn"` in
`configs/*.json`): a per-step MLP over a cost-annotated trajectory segment
sigma = (s_i, a_i, c_i, s'_i)_{i=1..m}, pooled by a learned softmax attention
gate (not a plain mean), followed by a tanh-bounded output head (not an L2
projection). `ContextDecoder` is COSTA's `nets/decoder.py::Decoder` used by
`OfflineMetaLearner` to predict the per-step cost bit from
concat(s, a, s', zeta) -- the auxiliary reconstruction term of Stage 1.
"""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from models.networks import MLP, default_init


class ContextEncoder(nn.Module):
    hidden_dims: Sequence[int]
    context_dim: int

    @nn.compact
    def __call__(self, segments: jnp.ndarray) -> jnp.ndarray:
        # segments: (B, m, seg_feat_dim)
        h = MLP(hidden_dims=tuple(self.hidden_dims), activate_final=True)(segments)  # (B, m, H)
        gate = nn.Dense(1, kernel_init=default_init(), name="Gate")(h)               # (B, m, 1)
        attn = nn.softmax(gate[..., 0], axis=1)                                      # (B, m)
        pooled = jnp.sum(attn[..., None] * h, axis=1)                                # (B, H)
        z = nn.Dense(self.context_dim, kernel_init=default_init(), name="CtxHead")(pooled)
        return nn.tanh(z)


class ContextDecoder(nn.Module):
    """Predicts the per-step cost bit from concat(s, a, s', zeta) (COSTA's
    `Decoder`, `nets/decoder.py`): an MLP with a sigmoid output, trained with a
    BCE reconstruction loss alongside the contrastive term (Sec. `sec:pooled-
    practical`, Stage 1)."""

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = MLP(hidden_dims=tuple(self.hidden_dims), activate_final=True)(x)
        out = nn.Dense(1, kernel_init=default_init(), name="Output")(h)
        return nn.sigmoid(out[..., 0])


# =============================================================================
# CoPDT context encoder (CDSRL/CoPDT/OSRL/osrl/algorithms/context_encoder.py)
#
# No discriminator, no regime labels and no relabeled counterfactual segments: the context
# is a *set* of transitions (s, a, s', r) of one task, tagged by their cost, and the encoder
# is trained purely by predicting next state / reward / cost from (zeta, s, a)
# (`CoPDTContextTrainer`).  The torch modules drop the safe / unsafe rows into two separate
# tensors; here the rows stay in place and a mask selects the branch, which is the same
# function (everything downstream is a permutation-invariant softmax pool) with static shapes.
# =============================================================================

class SafetyAwareEncoder(nn.Module):
    """x: (M, N, D) transitions, cost: (M, N) -> zeta: (M, context_dim) in (-1, 1).

    Safe (cost == 0) and unsafe (cost > 0) transitions go through separate MLPs.  With
    `simple_gate` they are pooled by one learned softmax gate; otherwise (CoPDT's default)
    a query is pooled from the safe features alone and cross-attends over all of them, so the
    code is read off *relative to* the safe behaviour."""
    hidden_dims: Sequence[int]
    output_dim: int
    simple_gate: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, cost: jnp.ndarray) -> jnp.ndarray:
        unsafe = cost > 0
        h_safe = MLP(tuple(self.hidden_dims), activate_final=True, name="SafeModel")(x)
        h_unsafe = MLP(tuple(self.hidden_dims), activate_final=True, name="UnsafeModel")(x)
        feat = jnp.where(unsafe[..., None], h_unsafe, h_safe)          # (M, N, H)
        H = feat.shape[-1]
        if self.simple_gate:
            gate = nn.Dense(1, kernel_init=default_init(), name="Gate")(feat)[..., 0]
            w = nn.softmax(gate, axis=-1)
            res = jnp.sum(w[..., None] * feat, axis=-2)
        else:
            qg = nn.Dense(1, kernel_init=default_init(), name="QueryGate")(feat)[..., 0]
            qg = jnp.where(unsafe, -1e9, qg)                            # query pooled over safe rows only
            qw = nn.softmax(qg, axis=-1)
            q = nn.Dense(H, kernel_init=default_init(), name="Query")(jnp.sum(qw[..., None] * feat, axis=-2))
            k = nn.Dense(H, kernel_init=default_init(), name="Key")(feat)
            v = nn.Dense(H, kernel_init=default_init(), name="Value")(feat)
            scores = jnp.einsum("...h,...nh->...n", q, k) / jnp.sqrt(H)
            res = jnp.einsum("...n,...nh->...h", nn.softmax(scores, axis=-1), v)
        return nn.tanh(nn.Dense(self.output_dim, kernel_init=default_init(), name="Out")(res))


class SimpleMlpEncoder(nn.Module):
    """The `simple_mlp` ablation: cost appended to every row, per-row MLP, mean pool."""
    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, cost: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, cost[..., None].astype(x.dtype)], axis=-1)
        h = MLP(tuple(self.hidden_dims), activate_final=True)(x)
        z = nn.tanh(nn.Dense(self.output_dim, kernel_init=default_init(), name="Out")(h))
        return z.mean(axis=-2)


class MultiHeadDecoder(nn.Module):
    """(zeta, s, a) -> (next-state latent, reward, P(cost > 0)); the three heads share a feature
    extractor.  Only used to train the encoder; it is discarded afterwards."""
    feature_hidden_dims: Sequence[int]
    decoder_hidden_dims: Sequence[int]
    state_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        f = MLP(tuple(self.feature_hidden_dims), activate_final=True, name="Features")(x)

        def head(out_dim, name):
            h = MLP(tuple(self.decoder_hidden_dims), activate_final=True, name=name + "Hidden")(f)
            return nn.Dense(out_dim, kernel_init=default_init(), name=name + "Out")(h)

        state_pred = head(self.state_dim, "State")
        reward_pred = head(1, "Reward")[..., 0]
        cost_prob = nn.sigmoid(head(1, "Cost")[..., 0])
        return state_pred, reward_pred, cost_prob
