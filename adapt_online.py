"""Online adaptation of a saved general policy (train_general.py) to one target goal, with
no target dataset: the analogue of COSTA's online_adaptation.py for IQLPooled.

Nothing is trained.  The policy, the critics and q_xi stay frozen; what adapts is the
constraint context zeta_hat that conditions them, which is re-inferred from the agent's
own online trajectories on the target env (no D_t, no D_s, no flow, no gradient steps):

  online trajectory k   rolled out under the incumbent context (trajectory 0 has none: it is
                        evaluate_pooled's burn-in under the safe fallback, then q_xi's context
                        from that segment is frozen for the rest of the episode)
  after trajectory k    the seg_len-step window with the most cost (the last window if the
                        trajectory was cost-free) is encoded by q_xi, and the incumbent
                        context is updated (--ctx_mode)
  after every --online_eval_every trajectories
                        the frozen policy is evaluated on the target under the incumbent
                        context, with no burn-in, for every eval budget

  --ctx_mode best    COSTA's "iid" selection: keep the context of the best trajectory so far,
                     a trajectory within budget (+ --safety_slack) beating one over it, then
                     the higher return among safe ones / the lower cost among unsafe ones
  --ctx_mode mean    average the contexts of all trajectories so far (COSTA's wo_iid)
  --ctx_mode latest  the context of the newest trajectory

The general policy with its usual per-episode burn-in (evaluate_pooled) is evaluated first as
the baseline; the curves are over online trajectory number, as in COSTA.

Writes to results/<group>/<seed>/online_g<goal>/.

    python adapt_online.py --config train_config.py:r --src_id 41 --tgt_goal 1 --train_seed 155 --eval_budgets "10,20,40,80"
"""
import json
import os
import sys

# keep the other adapt_*.py scripts' behaviour: jax must not grab the whole GPU
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.append('.')

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb
from absl import app, flags
from tqdm.auto import trange

from adapt_common import load_general_model, resolve_run
from train_gym import FLAGS, env_list, evaluate_pooled, make_multitask_ant_env, wrap_gym

flags.DEFINE_integer('online_episodes', 20, 'number of online trajectories collected on the target')
flags.DEFINE_enum('ctx_mode', 'best', ['best', 'mean', 'latest'], 'how the context is kept across trajectories (see the module docstring)')
flags.DEFINE_float('adapt_budget', -1.0, 'soft-constraint budget the online trajectories are rolled out under; <0 = the middle eval budget')
flags.DEFINE_float('safety_slack', 5.0, "--ctx_mode best: a trajectory counts as safe when its cost sum <= budget + slack (COSTA's safety_threshold + 5)")
flags.DEFINE_integer('online_eval_every', 1, 'evaluate the incumbent context after every this many online trajectories')
flags.DEFINE_integer('online_eval_episodes', 0, 'episodes per budget per evaluation; 0 = the saved config eval_episodes')


def step_cost(info):
    for k in ("cost", "violation", "cost_raw"):
        if k in info:
            return float(info[k])
    return 0.0


def rollout(agent, env, budget, max_steps, seg_len, ctx=None, seed=None):
    """One episode.  With `ctx`, that context is frozen from the first step; with None, the
    first `seg_len` steps are a burn-in under the zero context and the safe fallback, then the
    context q_xi infers from them is frozen (evaluate_pooled's deployment).  The budget is
    tracked by g(s, a, s', delta) after every post-burn-in step.

    Returns (agent, return, cost sum, steps (T, |s|+|a|+1+|s|) as concat(s, a, c, s'))."""
    obs, _ = env.reset(seed=seed)
    ret, cost, length, done, steps = 0.0, 0.0, 0, False, []
    context_dim = agent.context_dim

    def act(agent, obs):
        action, agent = agent.eval_actions(jnp.asarray(obs, jnp.float32))
        return np.asarray(action), agent

    def record(obs, action, c, nobs):
        steps.append(np.concatenate([np.asarray(obs, np.float32), np.asarray(action, np.float32),
                                     np.array([c], np.float32), np.asarray(nobs, np.float32)]))

    if ctx is None:
        agent = agent.set_context(np.zeros(context_dim, np.float32)).reset_budget(obs, 0.0)
        while len(steps) < seg_len and not done:
            action, agent = act(agent, obs)
            nobs, r, term, trunc, info = env.step(action)
            c = step_cost(info)
            record(obs, action, c, nobs)
            ret += r; cost += c; length += 1
            obs = nobs
            done = bool(term) or bool(trunc) or length >= max_steps
        if len(steps) == seg_len:
            ctx = np.asarray(agent.infer_context(jnp.asarray(np.stack(steps)[None]))[0])
    if ctx is not None:
        agent = agent.set_context(ctx).reset_budget(obs, float(budget))
        while not done:
            action, agent = act(agent, obs)
            nobs, r, term, trunc, info = env.step(action)
            agent = agent.step_budget(jnp.asarray(obs, jnp.float32), jnp.asarray(action, jnp.float32),
                                      jnp.asarray(nobs, jnp.float32))
            c = step_cost(info)
            record(obs, action, c, nobs)
            ret += r; cost += c; length += 1
            obs = nobs
            done = bool(term) or bool(trunc) or length >= max_steps
    return agent, ret, cost, np.stack(steps)


def informative_window(steps, m, cost_col):
    """The m-step window with the largest cost sum (the last one when nothing was costly);
    None if the trajectory is shorter than m."""
    if len(steps) < m:
        return None
    csum = np.convolve(steps[:, cost_col], np.ones(m), mode='valid')
    start = int(np.argmax(csum)) if csum.max() > 0 else len(steps) - m
    return steps[start:start + m]


def evaluate_context(agent, env, ctx, budgets, episodes, seed, max_steps, seg_len):
    """evaluate_pooled's metrics, but under a given frozen context and no burn-in."""
    results, rets, rates, sums, lens = {}, [], [], [], []
    for b in budgets:
        if hasattr(env, "set_target_cost"):
            env.set_target_cost(float(b))
        er, ec, el = [], [], []
        for _ in range(episodes):
            agent, ret, cost, steps = rollout(agent, env, b, max_steps, seg_len, ctx=ctx, seed=seed)
            er.append(ret); ec.append(cost); el.append(len(steps))
        mr, mc = float(np.mean(er)), float(np.mean(ec))
        results[f"budget_{b}/return"] = mr
        results[f"budget_{b}/cost"] = float(np.mean([c / max(l, 1) for c, l in zip(ec, el)]))
        results[f"budget_{b}/cost_sum"] = mc
        results[f"budget_{b}/episode_len"] = float(np.mean(el))
        if hasattr(env, "get_normalized_score"):
            n_ret, n_cost = env.get_normalized_score(mr, mc)
            results[f"budget_{b}/n_return"], results[f"budget_{b}/n_cost"] = float(n_ret), float(n_cost)
        rets.append(mr); rates.append(results[f"budget_{b}/cost"]); sums.append(mc); lens.append(results[f"budget_{b}/episode_len"])
    results.update({'return': float(np.mean(rets)), 'cost': float(np.mean(rates)), 'cost_sum': float(np.mean(sums)),
                    'episode_len': float(np.mean(lens))})
    for k in ('n_return', 'n_cost'):
        vals = [results[f"budget_{b}/{k}"] for b in budgets if f"budget_{b}/{k}" in results]
        if vals:
            results[k] = float(np.mean(vals))
    return results


def main(_):
    env_name = env_list[FLAGS.src_id]
    tgt_goal = FLAGS.tgt_goal
    group, seed, modeldir, cfg = resolve_run(env_name)

    budgets = (tuple(float(x) for x in FLAGS.eval_budgets.split(',') if x.strip())
               if FLAGS.eval_budgets else tuple(float(x) for x in cfg['eval_cost_thresholds']))
    pk, m = cfg['pooled_kwargs'], cfg['pooled_kwargs']['seg_len']
    adapt_budget = FLAGS.adapt_budget if FLAGS.adapt_budget >= 0 else budgets[len(budgets) // 2]
    eval_episodes = FLAGS.online_eval_episodes or cfg['eval_episodes']
    n_online = FLAGS.online_episodes

    outdir = os.path.join(modeldir, f"online_g{tgt_goal}")
    os.makedirs(outdir, exist_ok=True)
    wandb.init(project=FLAGS.project, name=f"{group}_{seed}_online_g{tgt_goal}", group=f"{group}_online",
               config={'online_episodes': n_online, 'ctx_mode': FLAGS.ctx_mode, 'adapt_budget': adapt_budget,
                       'safety_slack': FLAGS.safety_slack, 'tgt_goal': tgt_goal, 'budgets': budgets})

    # no D_t and no D_s: only the target env and the saved general model are needed
    tgt_env = wrap_gym(make_multitask_ant_env(env_name, tgt_goal), cost_limit=cfg['agent_kwargs']['cost_limit'])
    agent = load_general_model(modeldir, cfg, tgt_env.observation_space, tgt_env.action_space)
    obs_dim, act_dim = tgt_env.observation_space.shape[0], tgt_env.action_space.shape[0]
    cost_col = obs_dim + act_dim
    max_steps = getattr(getattr(tgt_env, "unwrapped", tgt_env), "_max_episode_steps", 1000)

    baseline = evaluate_pooled(seed, agent, tgt_env, cfg['eval_episodes'], seg_len=m,
                               context_dim=pk['context_dim'], cost_thresholds=budgets)
    print(f"[online] general policy (burn-in) on goal {tgt_goal}: "
          f"return={baseline['return']:.2f} cost={baseline['cost']:.4f} cost_sum={baseline['cost_sum']:.2f}")
    wandb.log({f"online/general/{k}": v for k, v in baseline.items()})

    # ---- online loop --------------------------------------------------------------
    ctx, best_key, ctx_sum, n_ctx = None, None, 0.0, 0
    ep_log, curve, contexts, acc_cost = [], [], [], 0.0
    for k in trange(n_online, desc='online adaptation'):
        # reset_task/seed once, then let the env's own rng vary the starts
        agent, ret, cost, steps = rollout(agent, tgt_env, adapt_budget, max_steps, m, ctx=ctx, seed=seed if k == 0 else None)
        acc_cost += cost
        ep_log.append({'trajectory': k + 1, 'return': float(ret), 'cost_sum': float(cost), 'length': len(steps),
                       'accumulated_cost': float(acc_cost), 'had_context': ctx is not None})

        window = informative_window(steps, m, cost_col)
        if window is not None:
            z = np.asarray(agent.infer_context(jnp.asarray(window[None]))[0])
            safe = cost <= adapt_budget + FLAGS.safety_slack
            key = (1, ret) if safe else (0, -cost)
            if FLAGS.ctx_mode == 'best':
                if best_key is None or key > best_key:
                    ctx, best_key = z, key
            elif FLAGS.ctx_mode == 'mean':
                ctx_sum, n_ctx = ctx_sum + z, n_ctx + 1
                ctx = ctx_sum / n_ctx
            else:
                ctx = z
        contexts.append(None if ctx is None else np.asarray(ctx))
        wandb.log({'online/trajectory': k + 1, **{f"online/train/{n}": v for n, v in ep_log[-1].items() if n != 'had_context'}})

        if ctx is not None and ((k + 1) % FLAGS.online_eval_every == 0 or k + 1 == n_online):
            info = evaluate_context(agent, tgt_env, ctx, budgets, eval_episodes, seed, max_steps, m)
            info['trajectory'] = k + 1
            curve.append(info)
            print(f"[online] after {k + 1} trajectories: return={info['return']:.2f} cost={info['cost']:.4f} "
                  f"cost_sum={info['cost_sum']:.2f} (online ep: return={ret:.2f} cost_sum={cost:.1f})")
            wandb.log({'online/trajectory': k + 1, **{f"online/eval/{n}": v for n, v in info.items() if n != 'trajectory'}})

    final = curve[-1] if curve else {}
    if final:
        wandb.log({k: v for k, v in final.items() if k != 'trajectory'})   # plain keys fill the runs table
        wandb.summary.update({k: v for k, v in final.items() if k != 'trajectory'})

    # ---- outputs ---------------------------------------------------------------------
    with open(os.path.join(outdir, 'eval.json'), 'w') as f:
        json.dump({'general': baseline, 'online': curve, 'train_episodes': ep_log,
                   'ctx_mode': FLAGS.ctx_mode, 'adapt_budget': adapt_budget}, f, indent=4, default=float)
    np.save(os.path.join(outdir, 'online_contexts.npy'),
            np.array([np.full(pk['context_dim'], np.nan) if c is None else c for c in contexts]))
    if curve:
        xs = [c['trajectory'] for c in curve]
        for name, ylabel in (('return', 'return'), ('cost', 'cost rate')):
            plt.plot(xs, [c[name] for c in curve], marker='o', label='online context')
            plt.axhline(baseline[name], color='gray', linestyle='--', label='general (burn-in)')
            plt.xlabel('online trajectory num'); plt.ylabel(ylabel)
            plt.title(f"Online adaptation on goal {tgt_goal} ({FLAGS.ctx_mode})")
            plt.legend(); plt.savefig(os.path.join(outdir, f"online_{name}.png")); plt.clf()
    print(f"[online] saved to {outdir}")


if __name__ == '__main__':
    app.run(main)
