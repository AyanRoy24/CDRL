"""Stage 0: train the per-family state / action auto-encoders that put every DSRL task into
one shared latent space (CoPDT: train_sa_encoder_share.py, here in JAX).

For each env family among --tasks (the variants of a family -- AntVel-v0/v1, CarButton1/2, ... --
share one encoder pair and are trained on their concatenated data):
  1. if the family's action dim != action_encode_dim: an action AE (tanh latent, noisy decode);
  2. the state AE with an inverse-dynamics head, on the (frozen) action latents.
The checkpoint with the lowest held-out loss is kept, written to <sa_dir>/<family>/.

    python train_sa_encoder.py --config train_config.py:r --tasks all
"""
import copy
import sys
import jax
import jax.numpy as jnp
import numpy as np
sys.path.append('.')

from absl import app
from tqdm.auto import trange

from copdt_common import FLAGS, Logger, to_dict
from dsrl_tasks import FAMILIES, load_dsrl_task, select_tasks
from models.agents.cbf.sa_encoder_trainer import (action_ae_eval, action_ae_step, create_action_ae,
                                                  create_state_ae, save_sa_encoder, state_ae_eval, state_ae_step)


def train_family(fam, tasks, p, sa, logger_enabled):
    seed, rng = p['seed'], np.random.default_rng(p['seed'])
    state_dim, action_dim = tasks[0].state_dim, tasks[0].action_dim
    print(f"\n===== {FAMILIES[fam]}: {[t.name for t in tasks]} (state {state_dim}, action {action_dim}) =====", flush=True)

    obs, nobs, act = [], [], []
    for t in tasks:
        _, d = load_dsrl_task(t, p['dsrl_kwargs']['cost_limit'])
        obs.append(d['observations']); nobs.append(d['next_observations']); act.append(d['actions'])
        del d
    obs = np.concatenate(obs).astype(np.float32)
    nobs = np.concatenate(nobs).astype(np.float32)
    act = np.clip(np.concatenate(act), -1 + 1e-5, 1 - 1e-5).astype(np.float32)
    n = len(obs)
    perm = rng.permutation(n)
    n_test = min(100_000, max(1, n // 50))
    test_idx, train_idx = perm[:n_test], perm[n_test:]          # small held-out slice picks the best epoch
    bs, steps = sa['batch_size'], sa['steps_per_epoch']
    logger = Logger(logger_enabled, FLAGS.project, f"sa_{FAMILIES[fam]}", "sa_encoder", {**sa, 'tasks': [t.name for t in tasks]})

    # ---------------------------------------------------------------- action AE
    train_action_ae = action_dim != sa['action_encode_dim']
    best_action_params = None
    if train_action_ae:
        a_def, a_ts = create_action_ae(seed, action_dim, sa['action_encode_dim'], sa['action_encoder_hidden_sizes'], sa['learning_rate'])
        key = jax.random.PRNGKey(seed + 1)
        best = np.inf
        noise = sa['noise_scale'] if sa['add_noise'] else 0.0
        for epoch in range(sa['action_epoch_num']):
            for _ in trange(steps, desc=f"action AE epoch {epoch + 1}", leave=False):
                key, k = jax.random.split(key)
                a_ts, loss = action_ae_step(a_ts, k, a_def, noise, jnp.asarray(act[rng.choice(train_idx, bs)]))
            eval_loss = float(action_ae_eval(a_ts.params, a_def, jnp.asarray(act[test_idx])))
            if eval_loss < best:
                best, best_epoch, best_action_params = eval_loss, epoch + 1, copy.deepcopy(jax.device_get(a_ts.params))
            logger.log({'action/train_loss': float(loss), 'action/eval_loss': eval_loss, 'action/best_epoch': best_epoch}, epoch + 1)
            print(f"[action AE] epoch {epoch + 1}: train {float(loss):.5f} eval {eval_loss:.5f} (best {best:.5f} @ {best_epoch})", flush=True)
        enc_a = jax.jit(lambda a: a_def.apply({'params': best_action_params}, a, method=type(a_def).encode))
        act_lat = np.concatenate([np.asarray(enc_a(jnp.asarray(act[i:i + 65536]))) for i in range(0, n, 65536)])
        act_dim_lat = sa['action_encode_dim']
    else:
        act_lat, act_dim_lat = act, action_dim

    # ---------------------------------------------------------------- state AE
    s_def, idm_def, s_ts = create_state_ae(seed, state_dim, act_dim_lat, sa['state_encode_dim'],
                                           sa['state_encoder_hidden_sizes'], sa['inverse_dynamics_model_hidden_sizes'],
                                           sa['learning_rate'])
    best, best_state_params = np.inf, None
    te = [jnp.asarray(x[test_idx]) for x in (obs, act_lat, nobs)]
    for epoch in range(sa['state_epoch_num']):
        for _ in trange(steps, desc=f"state AE epoch {epoch + 1}", leave=False):
            i = rng.choice(train_idx, bs)
            s_ts, info = state_ae_step(s_ts, s_def, idm_def, sa['idm_loss_weight'],
                                       jnp.asarray(obs[i]), jnp.asarray(act_lat[i]), jnp.asarray(nobs[i]))
        ev = {k: float(v) for k, v in state_ae_eval(s_ts.params, s_def, idm_def, sa['idm_loss_weight'], *te).items()}
        if ev['total_loss'] < best:
            best, best_epoch, best_state_params = ev['total_loss'], epoch + 1, copy.deepcopy(jax.device_get(s_ts.params['ae']))
        logger.log({**{f'state/train_{k}': float(v) for k, v in info.items()},
                    **{f'state/eval_{k}': v for k, v in ev.items()}, 'state/best_epoch': best_epoch}, epoch + 1)
        print(f"[state AE] epoch {epoch + 1}: eval total {ev['total_loss']:.5f} recon {ev['state_recon_loss']:.5f} "
              f"idm {ev['idm_loss']:.5f} (best {best:.5f} @ {best_epoch})", flush=True)

    cfg = {k: sa[k] for k in ('state_encode_dim', 'action_encode_dim', 'state_encoder_hidden_sizes',
                              'action_encoder_hidden_sizes', 'inverse_dynamics_model_hidden_sizes', 'idm_loss_weight')}
    cfg.update(state_dim=state_dim, action_dim=action_dim, tasks=[t.name for t in tasks], best_state_epoch=best_epoch)
    save_sa_encoder(f"{FLAGS.sa_dir}/{FAMILIES[fam]}", FAMILIES[fam], cfg, best_state_params, best_action_params)
    print(f"[sa_encoder] saved {FLAGS.sa_dir}/{FAMILIES[fam]}")
    logger.finish()


def main(_):
    p = FLAGS.config
    sa = to_dict(p['sa_encoder_kwargs'])
    tasks = select_tasks(FLAGS.tasks)
    for fam in sorted({t.family for t in tasks}):
        train_family(fam, [t for t in tasks if t.family == fam], to_dict(p), sa, FLAGS.wandb)


if __name__ == '__main__':
    app.run(main)
