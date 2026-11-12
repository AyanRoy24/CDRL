"""[Legacy COSTA multitask-ant pipeline; the DSRL/CoPDT entry point is train_general.py.]
Train the general safe policy over all the source goals and save it, with no
target input: the training half of the train/adapt split (COSTA's run_meta_cpq.py to
adapt_flow.py's online_adaptation.py).

Stage 1 trains the constraint-context encoder q_xi, stage 2 the context-conditioned
context-conditioned CBF (cbf_plain.py, fixed cost limit per map) on the pooled source regimes, via train_gym.call_main_transfer_pooled.
Everything lands in results/<env>_general/<seed>/, and config.json there is what
adapt_flow.py rebuilds the networks from.

--tgt_goal g keeps the target-side terms in training, as the pooled-only variant, saved
under results/<env>_pooled_tgt<g>/<seed>/.

    python train_general_costa.py --config train_config_costa.py:r --src_id 41 --src_goal "0,1,2" --config.seed 155
"""
import json
import os
import sys
import torch
sys.path.append('.')

from absl import app, flags

from train_gym import FLAGS, MULTITASK_ANT_ENVS, call_main_transfer_pooled, env_list, to_dict

FLAGS.set_default('tgt_goal', -1)   # -1 = the general policy; the flag is otherwise train_gym's
flags.mark_flag_as_required('config')


def main(_):
    print("GPU available", torch.cuda.is_available())
    p = FLAGS.config
    env = env_list[FLAGS.src_id]
    goals = [int(g) for g in FLAGS.src_goal]
    assert env in MULTITASK_ANT_ENVS, f"the pooled pipeline is only defined for {sorted(MULTITASK_ANT_ENVS)}"
    assert len(goals) > 1, "the general policy needs >1 --src_goal"
    tgt_goal = None if FLAGS.tgt_goal < 0 else FLAGS.tgt_goal

    p['project'] = FLAGS.project
    p['src_env'] = p['tgt_env'] = p['env_name'] = env
    p['src_goal'] = goals
    p['tgt_goal'] = p['goal_id'] = tgt_goal
    p['max_steps'] = FLAGS.max_steps
    p['agent_kwargs']['mode'] = FLAGS.mode
    p['agent_kwargs']['cost_tau'] = FLAGS.cost_tau
    p['agent_kwargs']['reward_tau'] = FLAGS.reward_tau
    p['agent_kwargs']['model_cls'] = 'CBFContext'   # plain hard-constraint CBF (cbf_plain.py) learner (was 'IQLPooled')
    if FLAGS.eval_budgets:
        p['eval_cost_thresholds'] = tuple(float(x) for x in FLAGS.eval_budgets.split(',') if x.strip())
    p['group'] = f"{env}_general" if tgt_goal is None else f"{env}_pooled_tgt{tgt_goal}"
    p['experiment_name'] = f"{p['group']}_{p['seed']}"

    modeldir = f"./results/{p['group']}/{p['seed']}"
    print(f"[general] seed {p['seed']} (random unless --config.seed is given) -> {modeldir}", flush=True)
    os.makedirs(modeldir, exist_ok=True)
    with open(os.path.join(modeldir, 'config.json'), 'w') as f:
        json.dump(to_dict(p), f, indent=4)
    call_main_transfer_pooled(p, goals)
    print(f"[general] saved to {modeldir}")


if __name__ == '__main__':
    app.run(main)
