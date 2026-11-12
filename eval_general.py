"""Evaluate a saved train_general.py model on chosen DSRL tasks (the "two maps", e.g. HopperVel-v0/-v1)
without training -- the CoPDT counterpart of OSRL's eval_mtcdt.py.

    python eval_general.py --run_dir results/copdt_general/336 \
        --eval_tasks "HopperVel-v0,HopperVel-v1" --eval_budgets "10,20,40,80" --eval_episodes 20

Everything (tasks, budgets, agent hyper-parameters, sa_dir/ctx_dir) is read from <run_dir>/config.json;
the flags only override.  --ckpt picks model<k>.pickle (default: the latest).  The protocol is the same
one train_general.py uses at eval time: per task and budget C, `eval_episodes` rollouts, each with a
fresh context zeta from that task's own data, decoded through the family's action decoder.
"""
import json
import os
import re
import sys
import jax.numpy as jnp
import numpy as np
sys.path.append('.')

from absl import app, flags

from copdt_common import FLAGS, LatentTaskData
from dsrl_tasks import select_tasks
from models.agents.cbf.copdt_context_trainer import load_context_encoder
from models.eval.evaluation_dsrl import _make_fns, evaluate_task
from models.agents.cbf.iql_pooled import IQLPooled

flags.DEFINE_string('run_dir', '', 'results/copdt_general/<seed> folder of the trained model')
flags.DEFINE_integer('ckpt', -1, 'model<k>.pickle to load; -1 = latest')
flags.DEFINE_string('eval_tasks', '', 'tasks to evaluate (names / family / indices, as --tasks); default: all tasks the run trained on')
flags.DEFINE_integer('eval_episodes', 0, 'episodes per (task, budget); 0 = the value in config.json')
flags.DEFINE_integer('eval_seed', -1, 'env/agent seed for evaluation; -1 = the training seed')
flags.mark_flag_as_required('run_dir')
flags.DEFINE_string('out', '', 'write the results json here (default <run_dir>/eval_<ckpt>.json)')


def make_agent(seed, latent_obs_dim, latent_act_dim, context_dim, ak, max_steps):
    """Same construction as train_general.make_agent (not imported: that module requires --config)."""
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
    with open(os.path.join(FLAGS.run_dir, 'config.json')) as f:
        p = json.load(f)
    ak, dk = p['agent_kwargs'], p['dsrl_kwargs']
    ckpt = FLAGS.ckpt
    if ckpt < 0:
        ckpt = max(int(m.group(1)) for m in map(lambda s: re.fullmatch(r'model(\d+)\.pickle', s), os.listdir(FLAGS.run_dir)) if m)
    seed = FLAGS.eval_seed if FLAGS.eval_seed >= 0 else p['seed']
    episodes = FLAGS.eval_episodes or p['eval_episodes']
    budgets = tuple(float(x) for x in FLAGS.eval_budgets.split(',') if x.strip()) or tuple(p['eval_cost_thresholds'])
    tasks = select_tasks(FLAGS.eval_tasks or ','.join(p['tasks']))
    trained = set(p['tasks'])
    assert all(t.name in trained for t in tasks), f"run was trained on {sorted(trained)} only"

    embed, ctx_cfg = load_context_encoder(p['ctx_dir'])
    ctx_size, ctx_rscale = ctx_cfg['context_size'], ctx_cfg['reward_scale']
    data = LatentTaskData(tasks, p['sa_dir'], dk['cost_limit'], reward_norm=dk['reward_norm'], seed=seed, keep_envs=True)

    agent = make_agent(seed, data.se, data.ae, ctx_cfg['context_encoding_dim'], ak, p['max_steps'])
    agent = agent.load(os.path.join(FLAGS.run_dir, f"model{ckpt}.pickle"))
    print(f"[eval] {FLAGS.run_dir} model{ckpt}: tasks {[t.name for t in tasks]}, budgets {budgets}, {episodes} episodes", flush=True)

    fns = {fam: _make_fns(c) for fam, c in data.codecs.items()}
    out = {}
    for k, t in enumerate(tasks):
        def context_fn(k=k):
            x, cost = data.sample_contexts([k], ctx_size, ctx_rscale)
            return np.asarray(embed(jnp.asarray(x), jnp.asarray(cost))[0])
        info = evaluate_task(seed, agent, data.envs[k], data.codecs[t.family], t.episode_len,
                             episodes, budgets, context_fn, fns=fns[t.family])
        out[t.name] = info
        print(f"{t.name}: n_return {info['n_return']:.3f} n_cost {info['n_cost']:.3f} | " +
              " ".join(f"C={int(b)}: ret {info[f'budget_{int(b)}/n_return']:.2f} cost {info[f'budget_{int(b)}/n_cost']:.2f}"
                       for b in budgets), flush=True)
    out['mean'] = {key: float(np.mean([out[t.name][key] for t in tasks])) for key in ('n_return', 'n_cost')}
    print("mean:", out['mean'])
    path = FLAGS.out or os.path.join(FLAGS.run_dir, f"eval_{ckpt}.json")
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[eval] saved {path}")


if __name__ == '__main__':
    app.run(main)
