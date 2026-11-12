"""JAX port of CDSRL/CoPDT/OSRL/osrl/algorithms/sa_encoder.py: the per-family state /
action auto-encoders (and the inverse-dynamics head that shapes the state latent) that put
every DSRL task into one shared (state_encode_dim, action_encode_dim) space.
"""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from models.networks.mlp import MLP


class StateAE(nn.Module):
    """State_AE: MLP encoder / mirrored MLP decoder, linear outputs."""
    state_dim: int
    encode_dim: int = 32
    hidden_sizes: Sequence[int] = (128, 128, 128)

    def setup(self):
        hs = tuple(self.hidden_sizes)
        self.encoder = MLP((*hs, self.encode_dim))
        self.decoder = MLP((*hs[::-1], self.state_dim))

    def __call__(self, state):
        z = self.encoder(state)
        return z, self.decoder(z)

    def encode(self, state):
        return self.encoder(state)

    def decode(self, z):
        return self.decoder(z)


class ActionAE(nn.Module):
    """Action_AE with tanh at both ends (osrl's `require_tanh=True`, the setting
    train_sa_encoder_share.py trains), so the action latent lives in [-1, 1]."""
    action_dim: int
    encode_dim: int = 2
    hidden_sizes: Sequence[int] = (128, 128, 128)

    def setup(self):
        hs = tuple(self.hidden_sizes)
        self.encoder = MLP((*hs, self.encode_dim))
        self.decoder = MLP((*hs[::-1], self.action_dim))

    def __call__(self, action, noise=None):
        z = nn.tanh(self.encoder(action))
        zn = z if noise is None else z + noise     # add_noise: decode from a perturbed latent
        return z, nn.tanh(self.decoder(zn))

    def encode(self, action):
        return nn.tanh(self.encoder(action))

    def decode(self, z):
        return nn.tanh(self.decoder(z))


class InverseDynamics(nn.Module):
    """inverse_dynamics_model: (z_s, z_s') -> action latent (tanh)."""
    action_dim: int
    hidden_sizes: Sequence[int] = (128, 128, 128)

    @nn.compact
    def __call__(self, z, next_z):
        x = jnp.concatenate([z, next_z], axis=-1)
        return nn.tanh(MLP((*tuple(self.hidden_sizes), self.action_dim))(x))
