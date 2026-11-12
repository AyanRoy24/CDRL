"""What adapt_flow.py and adapt_finetune.py share: the --group/--train_seed/--tgt_goal
flags, finding the train_general.py run to adapt, loading its networks, and building
the target dataset."""
import json
import os
import re

from absl import flags

from models.data.pooled_datasets import PooledSource, RegimeDataset
from train_gym import (FLAGS, ContextEncoderTrainer, _make_iql_pooled, ant_goal_dataset_path,
                       make_multitask_ant_env)

flags.DEFINE_string('group', '', 'results/<group> holding the general model; default <env>_general')
flags.DEFINE_integer('train_seed', -1, 'seed of the train_general.py run to adapt (its results/<group>/<seed> folder); -1 = the most recently trained one')
FLAGS.set_default('tgt_goal', None)   # absl's required check only fires on a None default
flags.mark_flags_as_required(['tgt_goal', 'config'])


def resolve_run(env_name):
    """(group, seed, modeldir, saved config.json) of the run to adapt."""
    group = FLAGS.group or f"{env_name}_general"
    seed = FLAGS.train_seed
    if seed < 0:
        runs = [d for d in os.listdir(f"./results/{group}")
                if d.isdigit() and os.path.isfile(f"./results/{group}/{d}/model1.pickle")] if os.path.isdir(f"./results/{group}") else []
        assert runs, f"no trained model in ./results/{group}: run train_general.py first"
        seed = int(max(runs, key=lambda d: os.path.getmtime(f"./results/{group}/{d}/model1.pickle")))
        print(f"[adapt] --train_seed not given: using the most recent run, seed {seed}")
    modeldir = f"./results/{group}/{seed}"
    assert os.path.isfile(os.path.join(modeldir, 'config.json')), f"no {modeldir}/config.json"
    with open(os.path.join(modeldir, 'config.json')) as f:
        return group, seed, modeldir, json.load(f)


def load_general_model(modeldir, cfg, obs_space, act_space):
    pk, seed, m = cfg['pooled_kwargs'], cfg['seed'], cfg['pooled_kwargs']['seg_len']
    ckpt = FLAGS.ckpt
    if ckpt < 0:
        found = [int(re.fullmatch(r'model(\d+)\.pickle', f).group(1)) for f in os.listdir(modeldir)
                 if re.fullmatch(r'model\d+\.pickle', f)]
        assert found, f"no model*.pickle in {modeldir}: run train_general.py first"
        ckpt = max(found)
    ctx_trainer = ContextEncoderTrainer.create(
        seed, obs_space.shape[0], act_space.shape[0], m,
        context_dim=pk['context_dim'], hidden_dims=tuple(pk['ctx_hidden_dims']),
        decoder_hidden_dims=tuple(pk['decoder_hidden_dims']),
        lr=pk['ctx_lr'], decoder_weight=pk['decoder_weight'],
    ).load(os.path.join(modeldir, f"context_encoder{ckpt}.pickle"))
    agent = _make_iql_pooled(seed, obs_space, act_space, ctx_trainer.encoder, pk, cfg['agent_kwargs'], m)
    print(f"[adapt] loaded checkpoint {ckpt} from {modeldir}")
    return agent.load(os.path.join(modeldir, f"model{ckpt}.pickle"))


def load_target(env_name, tgt_goal, cfg, data_kwargs, seed):
    """(PooledSource over the saved source goals, D_t).  D_t is that goal's own regime
    when the target is one of the source goals."""
    cost_scale = data_kwargs['cost_scale']
    pooled = PooledSource(env_name, cfg['src_goal'], data_kwargs['src_data'], cost_scale=cost_scale, seed=seed)
    if tgt_goal in cfg['src_goal']:
        return pooled, pooled.regimes[cfg['src_goal'].index(tgt_goal)]
    target = RegimeDataset(
        make_multitask_ant_env(env_name, tgt_goal), regime_id=-1, goal_id=tgt_goal,
        data_location=ant_goal_dataset_path(env_name, tgt_goal, data_kwargs.get('tgt_data', data_kwargs['src_data'])),
        cost_scale=cost_scale,
    )
    target.seed(seed + 1)
    return pooled, target
