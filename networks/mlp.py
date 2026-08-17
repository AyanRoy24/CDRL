from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
import flax
import distrax

# default_init = nn.initializers.xavier_uniform

# default_init = nn.initializers.uniform

def default_init(scale: float=2):
    return nn.initializers.orthogonal(scale**0.5)

def get_weight_decay_mask(params):
    flattened_params = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.unfreeze(params))

    def decay(k, v):
        if any([(key == 'bias' or 'Input' in key or 'Output' in key)
                for key in k]):
            return False
        else:
            return True

    return flax.core.frozen_dict.freeze(
        flax.traverse_util.unflatten_dict(
            {k: decay(k, v)
             for k, v in flattened_params.items()}))

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                # print(">>", x.dtype, type(x))
                x = nn.Dense(size, kernel_init=default_init())(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                x = self.activations(x)
        return x

class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    # droupout_rate: float = 0.25
    tanh_squash_distribution: bool  = False

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0,
        # training: bool = False
    ) -> distrax.Distribution:

        dims_including_final = (*self.hidden_dims, self.action_dim * 2,)
        means_and_logs = MLP(
            hidden_dims=dims_including_final,
            activate_final=False,
        )(observations)

        means, log_stds = jnp.split(means_and_logs, 2, axis=-1)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = distrax.MultivariateNormalDiag(loc=means,
                                                scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash_distribution:
            tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)
            return distrax.Transformed(base_dist, tanh_bijector)
        else:
            return base_dist