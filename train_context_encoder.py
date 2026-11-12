"""Stage 1: train the CoPDT constraint-context encoder over all --tasks (CoPDT:
train_context_encoder.py, here in JAX; needs the encoders from train_sa_encoder.py).

No discriminator and no regime labels: each update draws, for every task, `meta_batch_per_task`
context sets of `context_size` transitions (cost-tagged, in the shared latent space), encodes
each into zeta with the safety-aware encoder, and trains encoder + multi-head decoder to
predict next state / reward / cost from (zeta, s, a).  The encoder is written to --ctx_dir.

    python train_context_encoder.py --config train_config.py:r --tasks all
"""
import sys
import numpy as np
import jax.numpy as jnp
sys.path.append('.')

from absl import app
from tqdm.auto import trange

from copdt_common import FLAGS, LatentTaskData, Logger, to_dict
from dsrl_tasks import select_tasks
from models.agents.cbf.copdt_context_trainer import CoPDTContextTrainer


def main(_):
    p = FLAGS.config
    ck = to_dict(p['context_kwargs'])
    dk = to_dict(p['dsrl_kwargs'])
    tasks = select_tasks(FLAGS.tasks)
    steps = FLAGS.max_steps if FLAGS.max_steps > 0 else ck['update_steps']

    data = LatentTaskData(tasks, FLAGS.sa_dir, dk['cost_limit'], seed=p['seed'])
    ck['state_encoding_dim'], ck['action_encoding_dim'] = data.se, data.ae   # from the loaded auto-encoders
    cfg = {**ck, 'seed': p['seed'], 'tasks': [t.name for t in tasks], 'sa_dir': FLAGS.sa_dir}
    trainer = CoPDTContextTrainer.create(p['seed'], ck)
    logger = Logger(FLAGS.wandb, FLAGS.project, f"context_encoder_{p['seed']}", "context_encoder", cfg)

    T, mb = len(tasks), ck['meta_batch_per_task']
    task_ids = np.repeat(np.arange(T), mb)
    info = {}
    for step in trange(steps, desc='context encoder'):
        x, cost = data.sample_contexts(task_ids, ck['context_size'], ck['reward_scale'])
        trainer, info = trainer.update(jnp.asarray(x), jnp.asarray(cost))
        if (step + 1) % ck['eval_every'] == 0 or step == steps - 1:
            logged = {f"train/{k}": float(v) for k, v in info.items()}
            logger.log(logged, step)
            print(f"[context] step {step + 1}: " + " ".join(f"{k}={v:.4f}" for k, v in logged.items()), flush=True)
            trainer.save(FLAGS.ctx_dir, cfg)
    logger.finish()
    print(f"[context] saved to {FLAGS.ctx_dir}")


if __name__ == '__main__':
    app.run(main)
