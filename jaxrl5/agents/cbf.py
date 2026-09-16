import distrax
import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union, Any
import flax.linen as nn
from flax.core import FrozenDict
# import gym
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import flax
import pickle
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import GaussianPolicy, MixtureGaussianPolicy
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue,get_weight_decay_mask

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)
# def safe_expectile_loss(diff, expectile=0.8):
#     weight = jnp.where(diff < 0, expectile, (1 - expectile))
#     return weight * (diff**2)


def sample_deltabar(rng, shape, base_cost, max_cost):
    """Sample a budget (delta-bar) uniformly in [base_cost, max_cost].

    Mirrors CBF/bcrl: during training the reward critic / value / actor are
    conditioned on a random budget so that a single set of networks generalizes
    to any non-zero soft-constraint budget.
    """
    base_cost = jnp.clip(base_cost, 0.0, max_cost)
    uniform = jax.random.uniform(rng, shape, minval=0.0, maxval=1.0)
    scale = jnp.maximum(max_cost - base_cost, 0.0)
    return uniform * scale + base_cost


@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values
@partial(jax.jit, static_argnames=('safe_critic_fn'))
def compute_safe_q(safe_critic_fn, safe_critic_params, observations, actions):
    safe_q_values = safe_critic_fn({'params': safe_critic_params}, observations, actions)
    safe_q_values = safe_q_values.max(axis=0)
    return safe_q_values

class CBF(Agent):
    score_model: TrainState          # reward actor, budget-conditioned
    safe_score_model: TrainState     # safe (fallback) actor, NOT budget-conditioned
    critic: TrainState               # Q_r, budget-conditioned
    target_critic: TrainState
    value: TrainState                # V_r, budget-conditioned
    target_value: TrainState
    safe_critic: TrainState          # Q_h reachability barrier (hard constraint)
    safe_target_critic: TrainState
    safe_value: TrainState           # V_h reachability barrier
    cost_critic: TrainState          # Q_C cumulative discounted cost (for the budget)
    cost_target_critic: TrainState
    cost_value: TrainState           # V_C cumulative discounted cost
    delta_hat: jnp.ndarray           # running remaining budget used at eval time
    discount: float
    tau: float
    actor_tau: float
    reward_tau: float
    cost_tau: float
    cost_discount: float
    cost_expectile: float
    budget_max: float
    action_dim: int = struct.field(pytree_node=False)
    N : int = struct.field(pytree_node=False)
    reward_temperature: float
    cost_temperature: float
    cost_ub: float
    r_min: float
    mode:int = struct.field(pytree_node=False)  # 1: 'fisor', 2: 'add', 3: 'reach'
    tanh_scale: float
    num_components: int = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        cbf_lr: float = 1e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        reward_tau: float = 0.8,
        num_qs: int = 2,
        actor_weight_decay: Optional[float] = None,
        value_layer_norm: bool = False,
        critic_layer_norm: bool = True,
        reward_temperature: float = 3.0,
        cost_temperature: float = 5.0,
        cost_ub: float = 150.0,
        N: int = 64,
        decay_steps: Optional[int] = int(2e6),
        cost_tau: float = 0.2,
        r_min: float = -1.0,
        mode: int = 1,  # 1: 'fisor', 2: 'add', 3: 'reach'
        actor_tau: float = 0.001,
        cost_limit: float = 10,
        tanh_scale: float = 5,
        num_components: int = 5,
        cost_discount: float = 0.99,
        cost_expectile: float = 0.3,
        budget_max: float = 100.0,
    ):
        rng = jax.random.PRNGKey(seed)
        (rng, actor_key, safe_actor_key, critic_key, value_key,
         safe_critic_key, safe_value_key, cost_critic_key, cost_value_key) = jax.random.split(rng, 9)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        actor_def = GaussianPolicy(hidden_dims=actor_hidden_dims, action_dim=action_dim)
        # actor_def = MixtureGaussianPolicy(hidden_dims=actor_hidden_dims, action_dim=action_dim, num_components=num_components)
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        budget_sample = jnp.zeros((1, 1))

        # ---- reward actor (budget-conditioned) ----
        actor_params = actor_def.init(actor_key, observations, budget_sample)['params']
        actor_params = FrozenDict(actor_params)
        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr,
                                                       weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                                                       mask=get_weight_decay_mask,))

        # ---- safe fallback actor (NOT budget-conditioned) ----
        safe_actor_params = FrozenDict(actor_def.init(safe_actor_key, observations)['params'])
        safe_score_model = TrainState.create(apply_fn=actor_def.apply,
                                             params=safe_actor_params,
                                             tx=optax.adamw(learning_rate=actor_lr,
                                                            weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                                                            mask=get_weight_decay_mask,))


        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True,use_layer_norm=critic_layer_norm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        # reward critic is budget-conditioned
        critic_params = critic_def.init(critic_key, observations, actions, budget_sample)["params"]
        critic_params = FrozenDict(critic_params)
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # ---- reachability barrier critic Q_h (unconditioned, hard constraint) ----
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic_params = FrozenDict(safe_critic_params)
        safe_critic_optimiser = optax.adam(learning_rate=cbf_lr)
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=safe_critic_optimiser
        )
        safe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # ---- cumulative cost critic Q_C (unconditioned, for the soft budget) ----
        cost_critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        cost_critic_def = Ensemble(cost_critic_cls, num=num_qs)
        cost_critic_params = FrozenDict(cost_critic_def.init(cost_critic_key, observations, actions)["params"])
        cost_critic = TrainState.create(
            apply_fn=cost_critic_def.apply, params=cost_critic_params,
            tx=optax.adam(learning_rate=cbf_lr),
        )
        cost_target_critic = TrainState.create(
            apply_fn=cost_critic_def.apply, params=cost_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        # reward value is budget-conditioned
        value_params = value_def.init(value_key, observations, budget_sample)["params"]
        value_params = FrozenDict(value_params)
        value_optimiser = optax.adam(learning_rate=value_lr)
        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)
        target_value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # ---- reachability barrier value V_h (unconditioned) ----
        safe_value_def = StateValue(base_cls=value_base_cls)
        safe_value_params = FrozenDict(safe_value_def.init(safe_value_key, observations)["params"])
        safe_value = TrainState.create(apply_fn=safe_value_def.apply,
                                  params=safe_value_params,
                                  tx=value_optimiser)

        # ---- cumulative cost value V_C (unconditioned) ----
        cost_value_def = StateValue(base_cls=value_base_cls)
        cost_value_params = FrozenDict(cost_value_def.init(cost_value_key, observations)["params"])
        cost_value = TrainState.create(apply_fn=cost_value_def.apply,
                                       params=cost_value_params,
                                       tx=optax.adam(learning_rate=value_lr))


        return cls(
            actor=None, # Base class attribute
            score_model=score_model,
            safe_score_model=safe_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            target_value=target_value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            cost_critic=cost_critic,
            cost_target_critic=cost_target_critic,
            cost_value=cost_value,
            delta_hat=jnp.zeros((1,)),
            tau=tau,
            discount=discount,
            rng=rng,
            action_dim=action_dim,
            N=N,
            reward_tau=reward_tau,
            reward_temperature=reward_temperature,
            cost_temperature=cost_temperature,
            cost_tau=cost_tau,
            cost_ub=cost_ub,
            r_min=r_min,
            mode=mode,
            actor_tau=actor_tau,
            tanh_scale=tanh_scale,
            num_components=num_components,
            cost_discount=cost_discount,
            cost_expectile=cost_expectile,
            budget_max=budget_max,
        )

    def update_actor(agent, batch: DatasetDict, key: jnp.ndarray) -> Tuple[Agent, Dict[str, float]]:
        observations = batch["observations"]
        actions = batch["actions"]

        # cumulative-cost estimates for the state and the dataset action
        action_cost = agent.cost_target_critic.apply_fn(
            {"params": agent.cost_target_critic.params}, observations, actions
        ).max(axis=0)
        state_cost = agent.cost_value.apply_fn(
            {"params": agent.cost_value.params}, observations
        )

        # sample an absolute budget >= the action's expected cumulative cost
        deltabar = sample_deltabar(key, (observations.shape[0], 1), action_cost[..., None], agent.budget_max)

        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params}, observations, actions, deltabar
        )
        q = qs.min(axis=0)
        v = agent.value.apply_fn({"params": agent.value.params}, observations, deltabar)

        reward_adv = q - v
        reward_weights = jnp.clip(jnp.exp(reward_adv * agent.reward_temperature), 0, 100)

        # safety advantage: prefer actions with lower cumulative cost than the state baseline
        cost_adv = state_cost - action_cost
        cost_weights = jnp.clip(jnp.exp(cost_adv * agent.cost_temperature), 0, agent.cost_ub)

        # the actor sees the budget measured relative to the state's cost-to-go (the "overflow" budget)
        deltabar_state = jnp.clip(deltabar - state_cost[..., None], 0.0, agent.budget_max)

        def reward_actor_loss_fn(actor_params: FrozenDict[str, Any]):
            dist = agent.score_model.apply_fn({"params": actor_params}, observations, deltabar_state)
            log_probs = dist.log_prob(actions)
            actor_loss = -(log_probs * reward_weights).mean()
            return actor_loss, {"reward_actor_loss": actor_loss,
                                "reward_weights": reward_weights.mean(),
                                "reward_adv": reward_adv.mean()}

        grads, r_info = jax.grad(reward_actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        def safe_actor_loss_fn(actor_params: FrozenDict[str, Any]):
            dist = agent.safe_score_model.apply_fn({"params": actor_params}, observations)
            log_probs = dist.log_prob(actions)
            actor_loss = -(log_probs * cost_weights).mean()
            return actor_loss, {"safe_actor_loss": actor_loss,
                                "cost_weights": cost_weights.mean(),
                                "cost_adv": cost_adv.mean()}

        grads, s_info = jax.grad(safe_actor_loss_fn, has_aux=True)(agent.safe_score_model.params)
        safe_score_model = agent.safe_score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model, safe_score_model=safe_score_model)
        return agent, {**r_info, **s_info}

    @jax.jit
    def barrier_values(self, observations):
        return self.safe_value.apply_fn({"params": self.safe_value.params},observations)

    @jax.jit
    def cost_values(self, observations):
        return self.cost_value.apply_fn({"params": self.cost_value.params}, observations)

    @jax.jit
    def reset_budget(self, observation: jnp.ndarray, cost_threshold: jnp.ndarray):
        """Initialize the running budget at the start of an episode.

        delta_hat_0 = max(cost_threshold - V_C(s_0), 0)  (overflow budget, cf. bcrl Eq. 10).
        """
        obs_b = jnp.expand_dims(observation, axis=0)
        vc = self.cost_value.apply_fn({"params": self.cost_value.params}, obs_b)  # (1,)
        delta0 = jnp.maximum(jnp.reshape(cost_threshold, (1,)) - vc, 0.0)
        return self.replace(delta_hat=delta0)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        assert len(observations.shape) == 1
        observations_batch = jnp.expand_dims(observations, axis=0)

        state_cost = self.cost_value.apply_fn(
            {"params": self.cost_value.params}, observations_batch
        )  # (1,)

        # delta_hat is the running (overflow) budget for the current state
        deltabar_state = jnp.clip(self.delta_hat, 0.0, self.budget_max)[..., None]  # (1, 1)

        dist = self.score_model.apply_fn(
            {"params": self.score_model.params}, observations_batch, deltabar_state,
            temperature=0
        )
        safe_dist = self.safe_score_model.apply_fn(
            {"params": self.safe_score_model.params}, observations_batch,
            temperature=0
        )
        reward_action = dist.sample(seed=self.rng)
        safe_action = safe_dist.sample(seed=self.rng)

        # fall back to the safe actor once the budget is (near) exhausted
        use_safe = (deltabar_state[..., 0] < 1e-3)
        action = jnp.where(use_safe[..., None], safe_action, reward_action)

        # decrement the budget by the action's marginal cumulative cost (cf. bcrl Eq. 11)
        action_cost = self.cost_target_critic.apply_fn(
            {"params": self.cost_target_critic.params}, observations_batch, action
        ).max(axis=0)
        action_cost = jnp.maximum(action_cost, state_cost)
        new_delta_hat = jnp.maximum(self.delta_hat + (state_cost - action_cost), 0.0) / self.cost_discount

        return action.squeeze(), self.replace(delta_hat=new_delta_hat)


    def update_r(agent, batch: DatasetDict, key: jnp.ndarray) -> Tuple[Agent, Dict[str, float]]:
        """Budget-conditioned reward critic and value function update."""
        key_v, key_q = jax.random.split(key)
        observations = batch["observations"]
        actions = batch["actions"]
        next_observations = batch["next_observations"]
        raw_cost = batch["cost_raw"][..., None]

        action_cost = agent.cost_target_critic.apply_fn(
            {"params": agent.cost_target_critic.params}, observations, actions
        ).max(axis=0)  # (B,)

        # Step 1: Update V_r  (conditioned on a random absolute budget)
        deltabar_v = sample_deltabar(key_v, (observations.shape[0], 1), action_cost[..., None], agent.budget_max)
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params}, observations, actions, deltabar_v
        )
        q = jnp.min(qs, axis=0)

        def Vr_loss(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, observations, deltabar_v)
            loss = expectile_loss(q - v, agent.reward_tau).mean()  # tau -> 1
            return loss, {"v_r_loss": loss, "v_r": v.mean()}

        vr_grads, vr_info = jax.grad(Vr_loss, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=vr_grads)

        # Step 2: Update Q_r  (budget propagated through the Bellman recursion)
        next_state_cost = agent.cost_value.apply_fn(
            {"params": agent.cost_value.params}, next_observations
        )[..., None]
        next_state_cost = jnp.maximum(next_state_cost, (action_cost[..., None] - raw_cost) / agent.cost_discount)
        next_state_cost = jnp.maximum(next_state_cost, 0.0)

        deltabar_target = sample_deltabar(key_q, (observations.shape[0], 1), next_state_cost, agent.budget_max)
        deltabar = raw_cost + deltabar_target * agent.cost_discount  # absolute budget at this step

        next_vr = value.apply_fn({"params": value.params}, next_observations, deltabar_target)

        qh = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params}, observations, actions
        )
        qh_min = jnp.min(qh, axis=0)

        # feasible if reachability-safe (Q_h <= 0) OR the budget covers the expected cumulative cost
        budget_covers = deltabar[..., 0] >= action_cost
        feasible = jnp.logical_or(qh_min <= 0, budget_covers)

        safe_target = batch["rewards"] + agent.discount * batch["masks"] * next_vr
        unsafe_target = agent.r_min / (1 - agent.discount) - qh_min
        target_q = jnp.where(feasible, safe_target, unsafe_target)

        def Qr_loss(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, observations, actions, deltabar
            )
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"q_r_loss": loss, "q_r": qs.mean(), "feasible_frac": feasible.mean()}

        qr_grads, qr_info = jax.grad(Qr_loss, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=qr_grads)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(
            value=value,
            critic=critic,
            target_critic=target_critic
        )

        return new_agent, {**vr_info, **qr_info}


    def update_cost(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Cumulative discounted cost critic Q_C and value V_C (in-sample, optimistic)."""
        observations = batch["observations"]
        actions = batch["actions"]
        raw_cost = batch["cost_raw"]

        qcs = agent.cost_target_critic.apply_fn(
            {"params": agent.cost_target_critic.params}, observations, actions
        )
        qc = qcs.max(axis=0)

        def Vc_loss(cost_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vc = agent.cost_value.apply_fn({"params": cost_value_params}, observations)
            # expectile < 0.5 -> optimistic (low) cost-to-go estimate over dataset actions
            loss = expectile_loss(qc - vc, agent.cost_expectile).mean()
            return loss, {"vc_loss": loss, "v_c": vc.mean()}

        vc_grads, vc_info = jax.grad(Vc_loss, has_aux=True)(agent.cost_value.params)
        cost_value = agent.cost_value.apply_gradients(grads=vc_grads)

        next_vc = cost_value.apply_fn({"params": cost_value.params}, batch["next_observations"])
        target_qc = raw_cost + agent.cost_discount * batch["masks"] * next_vc

        def Qc_loss(cost_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcs = agent.cost_critic.apply_fn(
                {"params": cost_critic_params}, observations, actions
            )
            loss = ((qcs - target_qc) ** 2).mean()
            return loss, {"qc_loss": loss, "q_c": qcs.mean()}

        qc_grads, qc_info = jax.grad(Qc_loss, has_aux=True)(agent.cost_critic.params)
        cost_critic = agent.cost_critic.apply_gradients(grads=qc_grads)

        cost_target_critic_params = optax.incremental_update(
            cost_critic.params, agent.cost_target_critic.params, agent.tau
        )
        cost_target_critic = agent.cost_target_critic.replace(params=cost_target_critic_params)

        new_agent = agent.replace(
            cost_value=cost_value,
            cost_critic=cost_critic,
            cost_target_critic=cost_target_critic,
        )
        return new_agent, {**vc_info, **qc_info}


    def update_h(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Combined safety critic and value function update."""

        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"], batch["actions"]
        )
        qc = qcs.max(axis=0)  # max over ensemble

        def Vh_loss(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vh = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])
            loss = expectile_loss(qc - vh, agent.cost_tau).mean()
            return loss, {"vh_loss": loss, "v_h": vh.mean()}

        vh_grads, vh_info = jax.grad(Vh_loss, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=vh_grads)

        next_vh = safe_value.apply_fn(
            {"params": safe_value.params}, batch["next_observations"]
        )

        h_sa = batch["costs"]

        if agent.mode == 1:  # FISOR
            target_qh = (1 - agent.discount) * h_sa + agent.discount * jnp.maximum(h_sa, next_vh)
        elif agent.mode == 2:  # Value-as-Barrier (Additive Bellman)
            target_qh = h_sa + agent.discount * next_vh - (1 - agent.discount) * 0.6
        elif agent.mode == 3:  # Reachability Constrained RL
            target_qh = jnp.maximum(h_sa, next_vh)


        else:
            raise ValueError(f"Unknown CBF mode: {agent.mode}")



        def Qh_loss(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qhs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params},
                batch["observations"], batch["actions"]
            )
            # TD
            qh_loss = jnp.abs(qhs - target_qh).mean()

            # MSE
            # qh_loss = ((qhs - target_qh) ** 2).mean()

            def huber_loss(diff, delta=1.0):
                abs_diff = jnp.abs(diff)
                quadratic = jnp.minimum(abs_diff, delta)
                linear = abs_diff - quadratic
                return 0.5 * quadratic ** 2 + delta * linear

            # qh_loss = huber_loss(qhs - target_qh).mean()


            return qh_loss, {"qh_loss": qh_loss, "q_h": qhs.mean()}

        qh_grads, qh_info = jax.grad(Qh_loss, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=qh_grads)

        # Update target networks
        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)

        new_agent = agent.replace(
            safe_value=safe_value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic
        )

        return new_agent, {**vh_info, **qh_info}


    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        rng, key_r, key_a = jax.random.split(new_agent.rng, 3)
        new_agent, h_info = new_agent.update_h(batch)
        new_agent, c_info = new_agent.update_cost(batch)
        new_agent, r_info = new_agent.update_r(batch, key_r)
        new_agent, actor_info = new_agent.update_actor(batch, key_a)
        new_agent = new_agent.replace(rng=rng)
        info = {
            **h_info,
            **c_info,
            **r_info,
            **actor_info
        }
        return new_agent, info


    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent
