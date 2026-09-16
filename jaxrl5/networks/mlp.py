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
        self, observations: jnp.ndarray, budget: jnp.ndarray = None,
        temperature: float = 1.0,
        # training: bool = False
    ) -> distrax.Distribution:

        # Budget-conditioned policy: the remaining cost budget (delta-bar) is
        # concatenated to the observation, cf. CBF/bcrl budget-conditioned states.
        if budget is not None:
            observations = jnp.concatenate([observations, budget], axis=-1)

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
        

class MixtureGaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_components: int = 5  # Number of mixture components
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = False

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        # Output shape: (batch, num_components * (2 * action_dim + 1))
        # For each component: mean, log_std, and logit (weight)
        out_dim = self.num_components * (2 * self.action_dim + 1)
        outputs = MLP(
            hidden_dims=(*self.hidden_dims, out_dim),
            activate_final=False,
        )(observations)

        # Reshape to (batch, num_components, 2 * action_dim + 1)
        outputs = outputs.reshape((-1, self.num_components, 2 * self.action_dim + 1))
        means = outputs[..., :self.action_dim]
        log_stds = outputs[..., self.action_dim:2 * self.action_dim]
        logits = outputs[..., -1]

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        # Create a mixture of Gaussians
        base_dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        cat_dist = distrax.Categorical(logits=logits)
        mixture = distrax.MixtureSameFamily(
            mixture_distribution=cat_dist,
            components_distribution=base_dist
        )

        if self.tanh_squash_distribution:
            tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)
            return distrax.Transformed(mixture, tanh_bijector)
        else:
            return mixture