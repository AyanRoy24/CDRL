"""Deployment of the general budget-adaptive IQL policy on DSRL gymnasium tasks, in the CoPDT
protocol: no burn-in, no discriminator -- the context zeta is read from a set of
`context_size` transitions of the task's own offline data (CoPDT's "prompt"), the policy acts in
the shared latent space of the task family's frozen state / action auto-encoders, and every
soft-constraint budget in `cost_thresholds` (CoPDT: 10, 20, 40, 80) is rolled out on its own.
"""
from typing import Callable, Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import trange


def _step_cost(info):
    for k in ("cost", "violation", "cost_raw"):
        if k in info:
            return float(info[k])
    return 0.0


def _make_fns(codec):
    @jax.jit
    def act(agent, obs):
        lat_obs = codec.encode_state(obs)
        lat_act, agent = agent.eval_actions(lat_obs)
        lat_act = jnp.clip(lat_act, -1.0, 1.0)
        return lat_obs, lat_act, codec.decode_action(lat_act), agent

    @jax.jit
    def advance(agent, lat_obs, lat_act, next_obs):
        return agent.step_budget(lat_obs, lat_act, codec.encode_state(next_obs))

    return act, advance


def evaluate_task(seed, agent, env, codec, episode_len, num_episodes, cost_thresholds: Sequence[float],
                  context_fn: Callable[[], np.ndarray], fns=None) -> Dict[str, float]:
    """Per budget C: `num_episodes` episodes, each with a freshly drawn context `context_fn()`,
    the tracked budget reset to C and advanced by g(s, a, s', delta) after every step.  Reports
    return / cost (episode sum) / DSRL-normalised n_return, n_cost (n_cost relative to C)."""
    act, advance = fns if fns is not None else _make_fns(codec)
    low, high = np.asarray(env.action_space.low), np.asarray(env.action_space.high)
    results: Dict[str, float] = {}
    all_ret, all_cost, all_nret, all_ncost = [], [], [], []

    for C in cost_thresholds:
        env.set_target_cost(float(C))
        rets, costs = [], []
        for ep in trange(num_episodes, desc=f"  C={C}", leave=False):
            obs, _ = env.reset(seed=seed + ep)
            agent = agent.set_context(context_fn())
            agent = agent.reset_budget(obs, float(C))
            ret = cost = 0.0
            for _ in range(episode_len):
                lat_obs, lat_act, env_act, agent = act(agent, jnp.asarray(obs, jnp.float32))
                nobs, r, term, trunc, info = env.step(np.clip(np.asarray(env_act), low, high))
                agent = advance(agent, lat_obs, lat_act, jnp.asarray(nobs, jnp.float32))
                ret += float(r)
                cost += _step_cost(info)
                obs = nobs
                if term or trunc:
                    break
            rets.append(ret)
            costs.append(cost)
        mean_ret, mean_cost = float(np.mean(rets)), float(np.mean(costs))
        n_ret, n_cost = env.get_normalized_score(mean_ret, mean_cost)
        results[f"budget_{int(C)}/return"] = mean_ret
        results[f"budget_{int(C)}/cost"] = mean_cost
        results[f"budget_{int(C)}/n_return"] = float(n_ret)
        results[f"budget_{int(C)}/n_cost"] = float(n_cost)
        all_ret.append(mean_ret); all_cost.append(mean_cost)
        all_nret.append(float(n_ret)); all_ncost.append(float(n_cost))
    results.update({"return": float(np.mean(all_ret)), "cost": float(np.mean(all_cost)),
                    "n_return": float(np.mean(all_nret)), "n_cost": float(np.mean(all_ncost))})
    return results
