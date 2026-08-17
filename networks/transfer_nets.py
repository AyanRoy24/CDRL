from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from networks import MLP, default_init


class Encoder(nn.Module):
    """phi_enc : S x A -> Z, the coordinate system shared by both domains."""
    hidden_dims: Sequence[int]
    latent_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        h = MLP(hidden_dims=self.hidden_dims, activate_final=True)(inputs)
        return nn.Dense(self.latent_dim, kernel_init=default_init(), name="LatentDense")(h)


class Decoder(nn.Module):
    """d_dec : Z -> S x A, used only for the reconstruction loss in Step 1."""
    hidden_dims: Sequence[int]
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = MLP(hidden_dims=self.hidden_dims, activate_final=True)(z)
        return nn.Dense(self.obs_dim + self.action_dim, kernel_init=default_init(), name="ReconDense")(h)


class AutoEncoder(nn.Module):
    """Bundles phi_enc and d_dec so both are fit jointly by Step 1's reconstruction loss."""
    hidden_dims: Sequence[int]
    latent_dim: int
    obs_dim: int
    action_dim: int

    def setup(self):
        self.encoder = Encoder(hidden_dims=self.hidden_dims, latent_dim=self.latent_dim)
        self.decoder = Decoder(hidden_dims=self.hidden_dims, obs_dim=self.obs_dim, action_dim=self.action_dim)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray):
        z = self.encoder(observations, actions)
        recon = self.decoder(z)
        return z, recon

    def encode(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        return self.encoder(observations, actions)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)


class VelocityField(nn.Module):
    """v_eta : Z x [0, 1] -> Z, whose flow endpoint f_eta = psi_eta(., 1) is the source->target map."""
    hidden_dims: Sequence[int]
    latent_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        tau = jnp.broadcast_to(tau.reshape(-1, 1), (z.shape[0], 1)).astype(z.dtype)
        inputs = jnp.concatenate([z, tau], axis=-1)
        h = MLP(hidden_dims=self.hidden_dims, activate_final=True)(inputs)
        return nn.Dense(self.latent_dim, kernel_init=default_init(), name="VelocityDense")(h)
