"""What train_sa_encoder.py, train_context_encoder.py and train_general.py share: the CLI flags,
the logger, and `LatentTaskData` -- every selected DSRL task's offline data pushed through its
family's frozen state / action auto-encoder into the one shared latent space, with the two
samplers the later stages need (policy transitions, and CoPDT-style context sets).
"""
import json
import os

import numpy as np
from absl import flags
from ml_collections import ConfigDict, config_flags
import jax.numpy as jnp

from dsrl_tasks import load_dsrl_task, per_step_reward_scale
from models.agents.cbf.sa_encoder_trainer import load_codec

FLAGS = flags.FLAGS
flags.DEFINE_string('tasks', 'AntVel-v0,AntVel-v1', "'all', or comma list of task names / gym ids / family names / table indices "
                                    "(see dsrl_tasks.py), e.g. 'AntVel-v0,AntVel-v1' or 'AntVelocityGymnasium'")
flags.DEFINE_string('project', 'CoPDT-DSRL', 'wandb project')
flags.DEFINE_bool('wandb', True, 'log to wandb (off: stdout only)')
flags.DEFINE_string('sa_dir', 'results/sa_encoder', 'where train_sa_encoder.py writes / the others read the state/action AEs')
flags.DEFINE_string('ctx_dir', 'results/context_encoder/default', 'where train_context_encoder.py writes / train_general.py reads the context encoder')
flags.DEFINE_integer('max_steps', -1, 'override config.max_steps / config.context_kwargs.update_steps (-1 = config)')
flags.DEFINE_string('eval_budgets', '', 'comma-separated soft-constraint budgets for eval, e.g. "10,20,40,80" (default: config)')
config_flags.DEFINE_config_file('config', None, lock_config=False)


def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config


class Logger:
    """wandb if enabled (and importable), always stdout for the headline numbers."""

    def __init__(self, enabled, project, name, group, config):
        self.run = None
        if enabled:
            import wandb
            self.run = wandb.init(project=project, name=name, group=group, config=config)

    def log(self, data, step):
        if self.run is not None:
            self.run.log(data, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()


class LatentTaskData:
    """Offline data of `tasks`, encoded once into the shared latent space.

    Per transition (concatenated over tasks, `offsets[k]:offsets[k]+sizes[k]` is task k):
        obs / next_obs : (N, state_encode_dim)   z_s, z_s'
        act            : (N, action_encode_dim)  latent action (raw action for 2-d action tasks)
        rew            : (N,) raw reward         cost: (N,) raw cost in {0, 1}
        mask           : (N,) 0 only on true terminals (timeouts bootstrap -- the next obs is valid)
    Raw observations are dropped as soon as a task is encoded (26 tasks do not fit in RAM raw)."""

    def __init__(self, tasks, sa_dir, cost_limit, reward_norm=1.0, seed=0, keep_envs=False, chunk=65536):
        self.tasks = list(tasks)
        self.rng = np.random.default_rng(seed)
        self.envs = []
        self.codecs = {}          # family idx -> Codec
        self.sa_cfgs = {}
        self.reward_rscale = np.zeros(len(self.tasks), np.float32)   # per-task multiplier for the IQL reward
        parts = {k: [] for k in ("obs", "next_obs", "act", "rew", "cost", "mask")}
        sizes = []
        for k, t in enumerate(self.tasks):
            if t.family not in self.codecs:
                self.codecs[t.family], self.sa_cfgs[t.family] = load_codec(sa_dir, t.family_name, t.state_dim, t.action_dim)
            codec = self.codecs[t.family]
            env, d = load_dsrl_task(t, cost_limit)
            self.reward_rscale[k] = reward_norm * per_step_reward_scale(env, t)
            self.envs.append(env if keep_envs else None)
            enc = lambda fn, x: np.concatenate(
                [np.asarray(fn(jnp.asarray(x[i:i + chunk], jnp.float32))) for i in range(0, len(x), chunk)], axis=0)
            parts["obs"].append(enc(codec.encode_state, d["observations"]))
            parts["next_obs"].append(enc(codec.encode_state, d["next_observations"]))
            parts["act"].append(enc(codec.encode_action, np.clip(d["actions"], -1 + 1e-5, 1 - 1e-5)))
            parts["rew"].append(np.asarray(d["rewards"], np.float32))
            parts["cost"].append(np.asarray(d["costs"], np.float32))
            parts["mask"].append(1.0 - np.asarray(d["terminals"], np.float32))
            sizes.append(len(d["rewards"]))
            print(f"[data] {t.name}: {sizes[-1]} transitions, cost rate {parts['cost'][-1].mean():.4f}, "
                  f"reward x{self.reward_rscale[k]:.3f}", flush=True)
            del d, env
        for key, v in parts.items():
            setattr(self, key, np.concatenate(v, axis=0).astype(np.float32))
        self.sizes = np.asarray(sizes, np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.se = self.obs.shape[-1]
        self.ae = self.act.shape[-1]
        assert all(self.ae == c.action_encode_dim for c in self.codecs.values()), \
            "every family's action latent must have the same dim (action_encode_dim)"
        for k in range(len(self.tasks)):
            sl = slice(self.offsets[k], self.offsets[k] + self.sizes[k])
            assert (self.cost[sl] == 0).any(), f"{self.tasks[k].name} has no safe transition"

    # ---------------------------------------------------------------- samplers
    def _draw(self, task_ids, n):
        """(len(task_ids), n) global row indices, uniform within each task."""
        task_ids = np.asarray(task_ids)
        u = self.rng.random((len(task_ids), n))
        return self.offsets[task_ids][:, None] + (u * self.sizes[task_ids][:, None]).astype(np.int64)

    def sample_contexts(self, task_ids, n, ctx_reward_scale):
        """CoPDT context sets: for each entry of `task_ids`, n iid transitions of that task as
        x (M, n, 2se+ae+1) = (s, a, s', r * ctx_reward_scale) and cost (M, n); redrawn until the set
        holds at least one safe transition (the safety-aware encoder pools its query from those)."""
        idx = self._draw(task_ids, n)
        for _ in range(100):
            bad = ~(self.cost[idx] == 0).any(axis=1)
            if not bad.any():
                break
            idx[bad] = self._draw(np.asarray(task_ids)[bad], n)
        else:
            raise RuntimeError("could not draw a context with a safe transition")
        x = np.concatenate([self.obs[idx], self.act[idx], self.next_obs[idx],
                            (self.rew[idx] * ctx_reward_scale)[..., None]], axis=-1)
        return x.astype(np.float32), self.cost[idx]

    def sample_transitions(self, per_task):
        """`per_task` transitions of every task (rows grouped by task, task 0 first) as the
        batch dict IQLPooled.update expects; rewards are scaled per task, cost_raw is the raw cost."""
        T = len(self.tasks)
        idx = self._draw(np.arange(T), per_task)                       # (T, per)
        scale = self.reward_rscale[:, None]
        idx = idx.reshape(-1)
        return {
            "observations": self.obs[idx], "actions": self.act[idx], "next_observations": self.next_obs[idx],
            "rewards": (self.rew[idx].reshape(T, per_task) * scale).reshape(-1),
            "cost_raw": self.cost[idx], "masks": self.mask[idx],
        }


def find_ctx_cfg(ctx_dir):
    with open(os.path.join(ctx_dir, "config.json")) as f:
        return json.load(f)
