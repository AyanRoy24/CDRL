"""Stage 2: train the general safe policy over every DSRL task in --tasks (CoPDT's
train_mtcdt.py suite) with the budget-adaptive IQL (models/agents/cbf/iql_pooled.IQLPooled,
CBF/bcrl's bcrl_det.IQL), conditioned on the CoPDT context zeta.

    python train_sa_encoder.py      --config train_config.py:r --tasks all     # stage 0
    python train_context_encoder.py --config train_config.py:r --tasks all     # stage 1
    python train_general.py         --config train_config.py:r --tasks all     # stage 2 (this file)

Everything operates in the shared latent space of the frozen state / action auto-encoders: each
update draws transitions from every task, a fresh `context_size` context set per task is encoded
into zeta by the frozen context encoder, and the budget-conditioned critics / dual actor are fit
by expectile regression + advantage-weighted regression on (z_s, z_a, r, z_s', cost, zeta).  The
budget sampled for each transition is U[V_C-ish lower bound, delta_max], so one network serves
every budget; evaluation (CoPDT's protocol) rolls out C in --eval_budgets (default 10,20,40,80)
per task, with zeta read from that task's own data (no burn-in), decoding the latent action
through the family's action decoder.  Output: results/copdt_general/<seed>/{config.json,model<k>.pickle}.
"""
import json
import os
import sys
import jax.numpy as jnp
import numpy as np
sys.path.append('.')

from absl import app, flags
from tqdm.auto import trange

from copdt_common import FLAGS, LatentTaskData, Logger, to_dict
from dsrl_tasks import select_tasks
from models.agents.cbf.copdt_context_trainer import load_context_encoder
from models.agents.cbf.iql_pooled import IQLPooled
from models.eval.evaluation_dsrl import _make_fns, evaluate_task

flags.mark_flag_as_required('config')


def make_agent(seed, latent_obs_dim, latent_act_dim, context_dim, ak, max_steps):
    """IQLPooled over latent (state, action) spaces.  The context encoder is *not* part of the
    agent here (zeta is computed outside, from CoPDT context sets), so that field stays None."""
    import gymnasium as gym
    obs_space = gym.spaces.Box(-np.inf, np.inf, (latent_obs_dim,), np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, (latent_act_dim,), np.float32)
    return IQLPooled.create(
        seed, obs_space, act_space, context_encoder=None, context_dim=context_dim,
        actor_lr=ak['actor_lr'], critic_lr=ak['critic_lr'], value_lr=ak['value_lr'],
        cost_critic_lr=ak['cbf_lr'], cost_value_lr=ak['cbf_lr'],
        discount=ak['discount'], cost_discount=ak['cost_discount'],
        expectile=ak['reward_tau'], cost_expectile=ak['cost_expectile'],
        beta=ak['reward_temperature'], delta_max=ak['delta_max'],
        actor_weight_decay=ak['actor_weight_decay'],
        decay_steps=ak['decay_steps'] if ak['decay_steps'] is not None else max_steps,
        value_layer_norm=ak['value_layer_norm'],
    )


def main(_):
    p = to_dict(FLAGS.config)
    ak, dk, ck = p['agent_kwargs'], p['dsrl_kwargs'], p['context_kwargs']
    tasks = select_tasks(FLAGS.tasks)
    T = len(tasks)
    seed = p['seed']
    max_steps = FLAGS.max_steps if FLAGS.max_steps > 0 else p['max_steps']
    budgets = tuple(float(x) for x in FLAGS.eval_budgets.split(',') if x.strip()) or tuple(p['eval_cost_thresholds'])

    embed, ctx_cfg = load_context_encoder(FLAGS.ctx_dir)
    ctx_dim, ctx_size, ctx_rscale = ctx_cfg['context_encoding_dim'], ctx_cfg['context_size'], ctx_cfg['reward_scale']
    missing = [t.name for t in tasks if t.name not in ctx_cfg.get('tasks', [t.name])]
    if missing:
        print(f"[warn] the context encoder in {FLAGS.ctx_dir} was not trained on {missing}", flush=True)

    data = LatentTaskData(tasks, FLAGS.sa_dir, dk['cost_limit'], reward_norm=dk['reward_norm'], seed=seed, keep_envs=True)
    assert (data.se, data.ae) == (ctx_cfg['state_encoding_dim'], ctx_cfg['action_encoding_dim']), \
        "context encoder and state/action encoders disagree on the latent dims"

    agent = make_agent(seed, data.se, data.ae, ctx_dim, ak, max_steps)
    print(f"[general] {T} tasks, latent state {data.se} / action {data.ae}, context {ctx_dim}, "
          f"{int(data.sizes.sum())} transitions, eval budgets {budgets}", flush=True)

    p.update(tasks=[t.name for t in tasks], sa_dir=FLAGS.sa_dir, ctx_dir=FLAGS.ctx_dir,
             eval_cost_thresholds=list(budgets), max_steps=max_steps)
    group = "copdt_general"
    modeldir = f"./results/{group}/{seed}"
    os.makedirs(modeldir, exist_ok=True)
    with open(os.path.join(modeldir, 'config.json'), 'w') as f:
        json.dump(p, f, indent=4)
    logger = Logger(FLAGS.wandb, FLAGS.project, f"{group}_{seed}", group, p)

    fns = {fam: _make_fns(c) for fam, c in data.codecs.items()}
    per_task = max(1, 2 * p['batch_size'] // T)
    save_time = 1

    def context_fn(k):
        def draw():
            x, cost = data.sample_contexts([k], ctx_size, ctx_rscale)
            return np.asarray(embed(jnp.asarray(x), jnp.asarray(cost))[0])
        return draw

    def evaluate(step):
        results, per_task_res = {}, {}
        for k, t in enumerate(tasks):
            info = evaluate_task(seed, agent, data.envs[k], data.codecs[t.family], t.episode_len,
                                 p['eval_episodes'], budgets, context_fn(k), fns=fns[t.family])
            per_task_res[t.name] = info
            results.update({f"{t.name}/{key}": v for key, v in info.items()})
            print(f"[eval {step}] {t.name}: n_return {info['n_return']:.3f} n_cost {info['n_cost']:.3f} "
                  f"| " + " ".join(f"C={int(b)}: ret {info[f'budget_{int(b)}/n_return']:.2f} "
                                   f"cost {info[f'budget_{int(b)}/n_cost']:.2f}" for b in budgets), flush=True)
        for key in ("n_return", "n_cost", "return", "cost"):
            results[f"mean/{key}"] = float(np.mean([r[key] for r in per_task_res.values()]))
        return results

    for i in trange(max_steps, smoothing=0.1, desc=group):
        batch = {k: jnp.asarray(v) for k, v in data.sample_transitions(per_task).items()}
        x, cost = data.sample_contexts(np.arange(T), ctx_size, ctx_rscale)
        zeta = embed(jnp.asarray(x), jnp.asarray(cost))                  # (T, ctx_dim), frozen encoder
        ctx = jnp.repeat(zeta, per_task, axis=0)                          # rows are grouped by task
        agent, info = agent.update(batch, None, ctx, None)

        if i % p['log_interval'] == 0:
            logger.log({f"train/{k}": float(v) for k, v in info.items()}, i)
        if i % p['eval_interval'] == 0 and i > 0 or i == max_steps - 1:
            res = evaluate(i)
            logger.log(res, i)
            print({k: v for k, v in res.items() if k.startswith('mean/')}, flush=True)
            agent.save(modeldir, save_time)
            save_time += 1
    logger.finish()
    print(f"[general] saved to {modeldir}")


if __name__ == '__main__':
    app.run(main)
