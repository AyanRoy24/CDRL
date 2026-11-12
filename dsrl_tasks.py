"""The DSRL task suite CoPDT trains on (CDSRL/CoPDT/OSRL/examples/train/train_mtcdt.py), and
the loader that turns one task into (env, offline dataset).

26 tasks over 13 env families; the variants of one family (e.g. AntVelocity v0/v1, or
CarButton1/2) share one state/action auto-encoder, exactly as train_sa_encoder_share.py groups
them.  Everything per-task that train_mtcdt.py hard-codes lives in the one table below.
"""
from dataclasses import dataclass
from typing import List


# family order is train_mtcdt.py's `task_envs` / `env_state_dims` / `env_action_dims` order
FAMILIES = [
    "PointButtonGymnasium", "PointCircleGymnasium", "PointGoalGymnasium", "PointPushGymnasium",
    "HalfCheetahVelocityGymnasium", "HopperVelocityGymnasium",
    "CarButtonGymnasium", "CarCircleGymnasium", "CarGoalGymnasium", "CarPushGymnasium",
    "AntVelocityGymnasium", "SwimmerVelocityGymnasium", "Walker2dVelocityGymnasium",
]
FAMILY_STATE_DIMS = [76, 28, 60, 76, 17, 11, 88, 40, 72, 88, 27, 8, 17]
FAMILY_ACTION_DIMS = [2, 2, 2, 2, 6, 3, 2, 2, 2, 2, 8, 2, 6]

_GYM_IDS = [
    "OfflinePointButton1Gymnasium-v0", "OfflinePointButton2Gymnasium-v0",
    "OfflinePointCircle1Gymnasium-v0", "OfflinePointCircle2Gymnasium-v0",
    "OfflinePointGoal1Gymnasium-v0", "OfflinePointGoal2Gymnasium-v0",
    "OfflinePointPush1Gymnasium-v0", "OfflinePointPush2Gymnasium-v0",
    "OfflineHalfCheetahVelocityGymnasium-v0", "OfflineHalfCheetahVelocityGymnasium-v1",
    "OfflineHopperVelocityGymnasium-v0", "OfflineHopperVelocityGymnasium-v1",
    "OfflineCarButton1Gymnasium-v0", "OfflineCarButton2Gymnasium-v0",
    "OfflineCarCircle1Gymnasium-v0", "OfflineCarCircle2Gymnasium-v0",
    "OfflineCarGoal1Gymnasium-v0", "OfflineCarGoal2Gymnasium-v0",
    "OfflineCarPush1Gymnasium-v0", "OfflineCarPush2Gymnasium-v0",
    "OfflineAntVelocityGymnasium-v0", "OfflineAntVelocityGymnasium-v1",
    "OfflineSwimmerVelocityGymnasium-v0", "OfflineSwimmerVelocityGymnasium-v1",
    "OfflineWalker2dVelocityGymnasium-v0", "OfflineWalker2dVelocityGymnasium-v1",
]
_NAMES = [
    "PointButton1", "PointButton2", "PointCircle1", "PointCircle2", "PointGoal1", "PointGoal2",
    "PointPush1", "PointPush2", "HalfCheetahVel-v0", "HalfCheetahVel-v1", "HopperVel-v0", "HopperVel-v1",
    "CarButton1", "CarButton2", "CarCircle1", "CarCircle2", "CarGoal1", "CarGoal2", "CarPush1", "CarPush2",
    "AntVel-v0", "AntVel-v1", "SwimmerVel-v0", "SwimmerVel-v1", "Walker2dVel-v0", "Walker2dVel-v1",
]
_FAMILY_OF = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12]
_EPISODE_LENS = [1000, 1000, 500, 500, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                 1000, 1000, 500, 500, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]


@dataclass(frozen=True)
class Task:
    idx: int            # position in the full 26-task table
    name: str
    gym_id: str
    family: int         # index into FAMILIES
    state_dim: int
    action_dim: int
    episode_len: int

    @property
    def family_name(self):
        return FAMILIES[self.family]


TASKS: List[Task] = [
    Task(i, _NAMES[i], _GYM_IDS[i], _FAMILY_OF[i], FAMILY_STATE_DIMS[_FAMILY_OF[i]],
         FAMILY_ACTION_DIMS[_FAMILY_OF[i]], _EPISODE_LENS[i])
    for i in range(len(_GYM_IDS))
]


def select_tasks(spec: str) -> List[Task]:
    """'all', or a comma list of task names ('AntVel-v0'), gym ids, family names
    ('AntVelocityGymnasium', every variant) or full-table indices ('20')."""
    if spec.strip() in ("", "all"):
        return list(TASKS)
    picked = []
    for tok in [s.strip() for s in spec.split(",") if s.strip()]:
        hit = [t for t in TASKS if tok in (t.name, t.gym_id, t.family_name) or tok == str(t.idx)]
        assert hit, f"unknown task '{tok}'; names: {[t.name for t in TASKS]}, families: {FAMILIES}"
        picked += [t for t in hit if t not in picked]
    return sorted(picked, key=lambda t: t.idx)


def _alias_safe_velocity():
    """CoPDT's DSRL fork imports `safety_gymnasium.tasks.safe_velocity.*`, but the safety_gymnasium
    installed here ships the same task files as `safety_gymnasium.tasks.safety_velocity.*`.  Alias
    the former to the latter so the fork imports without editing either package."""
    import importlib
    import sys
    try:
        importlib.import_module("safety_gymnasium.tasks.safe_velocity")
        return
    except ImportError:
        pass
    src = "safety_gymnasium.tasks.safety_velocity"
    pkg = importlib.import_module(src)
    sys.modules["safety_gymnasium.tasks.safe_velocity"] = pkg
    for agent in ("ant", "half_cheetah", "hopper", "swimmer", "walker2d"):
        for v in ("v0", "v1"):
            name = f"safety_{agent}_velocity_{v}"
            try:
                sys.modules[f"safety_gymnasium.tasks.safe_velocity.{name}"] = importlib.import_module(f"{src}.{name}")
            except ImportError:
                pass


def load_dsrl_task(task: Task, cost_limit: float):
    """(env, data): the registered DSRL gymnasium env and its pre-processed offline dataset
    (train_mtcdt.py / train_sa_encoder_share.py load it the same way, with no outlier / noise /
    inpainting / density modification)."""
    _alias_safe_velocity()
    import dsrl  # noqa: F401  (registers the Offline*Gymnasium envs)
    import gymnasium as gym
    env = gym.make(task.gym_id)
    env.set_target_cost(cost_limit)
    data = env.get_dataset()
    data = env.pre_process_data(data, None, None, None, None, 1.0,
                                cbins=None, rbins=None, max_npb=None, min_npb=None)
    assert data["observations"].shape[-1] == task.state_dim, \
        f"{task.name}: dataset obs dim {data['observations'].shape[-1]} != table {task.state_dim}"
    assert data["actions"].shape[-1] == task.action_dim, \
        f"{task.name}: dataset action dim {data['actions'].shape[-1]} != table {task.action_dim}"
    return env, data


def per_step_reward_scale(env, task: Task) -> float:
    """Multiplier that makes an expert's per-step reward ~1 in every task, so the one shared
    critic is not dominated by e.g. HalfCheetah's ~3/step against PointButton's ~0.05/step."""
    max_ret = float(getattr(env, "max_episode_reward", 0.0) or 0.0)
    return task.episode_len / max(max_ret, 1e-3)
