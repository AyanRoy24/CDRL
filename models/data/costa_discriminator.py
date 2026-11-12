"""CDSRL/COSTA/COSTA's per-task cost discriminators, used as the cost relabelers of the
context encoder's counterfactual objective (`OfflineMetaLearner.create_negative_cost` /
`create_positive_cost`).

COSTA's `run_discriminator.py` trains one `EnsembleDiscriminatorModel` per task, saved as
`<costa_dir>/run/<env>_<goal>/discriminator/best_model.pt`, mapping concat(s, a, s') to the
probability that the transition is costly under that task's cost function.  Relabeling a
transition with task h's discriminator (ensemble mean, thresholded at 0.5, as COSTA does)
gives the cost bit the same (s, a, s') would have under regime h.  That is what stops the
encoder from telling regimes apart by their state/action statistics alone.

The checkpoints are whole pickled torch modules, so COSTA's `modules` and `nets` packages
have to be importable; they are only torch code and are appended to the END of sys.path, so
nothing this repo already provides is shadowed.  Labels are computed once per (regime,
discriminator) pair and cached under ./relabel_cache/.
"""

import hashlib
import importlib
import os
import sys

import numpy as np

from models.env.multitask_ant import dataset_env_name

CACHE_DIR = "./relabel_cache"


def _import_costa(costa_dir):
    costa_dir = os.path.abspath(costa_dir)
    assert os.path.isdir(costa_dir), f"costa_dir {costa_dir} does not exist"
    for pkg in ("modules", "nets"):
        loaded = sys.modules.get(pkg)
        if loaded is not None:
            where = os.path.abspath(os.path.dirname(getattr(loaded, "__file__", "") or ""))
            assert where.startswith(costa_dir), \
                f"a different '{pkg}' package ({where}) is already imported; COSTA's cannot be loaded next to it"
    if costa_dir not in sys.path:
        sys.path.append(costa_dir)
    importlib.import_module("modules.discriminator_module")


def discriminator_path(costa_dir, env_name, goal_id):
    return os.path.join(costa_dir, "run", f"{dataset_env_name(env_name)}_{goal_id}", "discriminator", "best_model.pt")


def load_discriminators(costa_dir, env_name, goal_ids, device="cpu"):
    """{goal_id: EnsembleDiscriminatorModel} for every goal id, in eval mode."""
    import torch
    paths = {g: discriminator_path(costa_dir, env_name, g) for g in goal_ids}
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"no COSTA discriminator for {env_name} goals {list(goal_ids)}: missing {missing}. COSTA only "
            f"trains them for goals 0,1,2 of some envs (run_discriminator.py); set pooled_kwargs.ctx_relabel=False "
            f"to train the context encoder without the counterfactual relabeling.")
    _import_costa(costa_dir)
    models = {}
    for g, p in paths.items():
        m = torch.load(p, map_location=device, weights_only=False)
        models[g] = m.to(device).eval()
    return models, paths


def label_transitions(model, obs, act, nobs, device="cpu", chunk=65536):
    """Cost bit (float32 0/1, shape (N,)) the discriminator assigns to each (s, a, s')."""
    import torch
    out = np.empty(len(obs), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(obs), chunk):
            x = np.concatenate([obs[i:i + chunk], act[i:i + chunk], nobs[i:i + chunk]], axis=-1)
            p = model(torch.as_tensor(x, dtype=torch.float32, device=device)).mean(0)[:, 0]
            out[i:i + chunk] = (p >= 0.5).float().cpu().numpy()
    return out


def _cache_file(data_location, disc_path, n):
    def stamp(p):
        st = os.stat(p)
        return f"{os.path.abspath(p)}:{st.st_size}:{int(st.st_mtime)}"
    key = hashlib.md5(f"{stamp(data_location)}|{stamp(disc_path)}|{n}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{key}.npy")


def relabel_regime(regime, discriminators, paths, device="cpu"):
    """Attach to `regime` the counterfactual cost bits of every OTHER regime's discriminator
    (`regime.cf_costs[h]`), and log how often its own discriminator reproduces the recorded
    cost (a layout check: it should be ~1)."""
    d = regime.dataset_dict
    obs, act, nobs = d["observations"], d["actions"], d["next_observations"]
    os.makedirs(CACHE_DIR, exist_ok=True)
    for h, model in discriminators.items():
        if h == regime.goal_id:
            sub = np.sort(np.random.default_rng(0).choice(len(obs), min(len(obs), 50000), replace=False))
            lab = label_transitions(model, obs[sub], act[sub], nobs[sub], device)
            agree = float(np.mean(lab == (np.asarray(d["cost_raw"])[sub] > 0)))
            print(f"[relabel] goal {regime.goal_id}: own discriminator agrees with the recorded cost on {agree:.3f} "
                  f"of transitions", flush=True)
            if agree < 0.9:
                print(f"[relabel] WARNING: goal {regime.goal_id}: the discriminator does not match this dataset's "
                      f"cost labels (observation layout?)", flush=True)
            continue
        f = _cache_file(regime.data_location, paths[h], len(obs))
        if os.path.isfile(f):
            lab = np.load(f).astype(np.float32)
        else:
            lab = label_transitions(model, obs, act, nobs, device)
            np.save(f, lab.astype(np.uint8))
        regime.set_counterfactual_costs(h, lab)
        print(f"[relabel] goal {regime.goal_id} relabeled by goal {h}'s discriminator: "
              f"{lab.mean():.3f} costly (recorded {np.mean(np.asarray(d['cost_raw']) > 0):.3f})", flush=True)
