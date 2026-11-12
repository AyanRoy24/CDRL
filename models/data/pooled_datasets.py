"""Pooled multi-source data for Part III (Sec. `sec:pooled`).

The source side is several CoSTA multitask-ant goals that share `S, A, T, r` and
differ only in the per-goal cost function `c^(k)` (the safe-corridor definition in
`ant_dir_safe.py` &co.).  There is no recorded task label, so the goal id is used
*only* during Stage-1 training to build contrastive positives/negatives -- it is
never fed to the critics or the policy.

Each regime `k` is one `<env_name>_<goal_id>.hdf5` buffer, loaded exactly as the
single-source transfer path loads it (`DSRLDataset`), so it already carries both
the binarised barrier label `costs = h(s,a) in {+cost_scale, -1}` and the raw
non-negative per-step cost `cost_raw = c(s,a) >= 0` that the additive cost critic
`Q_C` and the context encoder `q_xi` consume.  On top of that this module adds the
trajectory structure (episode slices from the terminal/timeout signal) needed to
draw contiguous cost-annotated segments.
"""

import numpy as np

from models.data.dsrl_datasets import DSRLDataset
from models.env.multitask_ant import make_multitask_ant_env, ant_goal_dataset_path


def _episode_slices(dones):
    slices, start = [], 0
    for i, d in enumerate(dones):
        if d:
            slices.append((start, i + 1))
            start = i + 1
    if start < len(dones):
        slices.append((start, len(dones)))
    return slices


class RegimeDataset(DSRLDataset):
    """One constraint regime (one goal id): a per-goal hdf5 buffer + the
    trajectory structure needed to draw contiguous cost-annotated segments."""

    def __init__(self, env, regime_id, goal_id, data_location, cost_scale=1.0):
        super().__init__(env, cost_scale=cost_scale, data_location=data_location)
        self.regime_id = regime_id
        self.goal_id = goal_id
        self.data_location = data_location
        # counterfactual cost bits (N,) of the OTHER regimes' cost functions on this regime's
        # transitions, keyed by their goal id (set_counterfactual_costs, models/data/costa_discriminator.py)
        self.cf_costs = {}

        # DSRLDataset merges terminals|timeouts into `masks` (0 at an episode
        # boundary) and drops the raw fields; recover the boundary signal here.
        dones = (np.asarray(self.dataset_dict["masks"]) < 0.5)
        self._episodes = _episode_slices(dones)
        self._valid_start_cache = {}

    def set_counterfactual_costs(self, goal_id, costs):
        assert len(costs) == len(self.dataset_dict["observations"])
        self.cf_costs[int(goal_id)] = np.asarray(costs, dtype=np.float32)

    def _cost_signal(self, cost_goal=None):
        """Per-transition cost the windows are anchored on / carry: the recorded cost_raw, or the
        cost bit another regime's discriminator assigns to the same transition."""
        if cost_goal is None or int(cost_goal) == self.goal_id:
            return np.asarray(self.dataset_dict["cost_raw"])
        return self.cf_costs[int(cost_goal)]

    def _valid_starts(self, m, cost_goal=None):
        key = (m, None if cost_goal is None or int(cost_goal) == self.goal_id else int(cost_goal))
        if key not in self._valid_start_cache:
            craw = self._cost_signal(cost_goal)
            alls, anchored = [], []
            for s, e in self._episodes:
                for i in range(s, e - m + 1):
                    alls.append(i)
                    if craw[i : i + m].sum() > 0.0:
                        anchored.append(i)
            self._valid_start_cache[key] = (
                np.asarray(alls or [0], dtype=np.int64),
                np.asarray(anchored, dtype=np.int64),
            )
        return self._valid_start_cache[key]

    def segment_feature_dim(self):
        o = self.dataset_dict["observations"].shape[-1]
        a = self.dataset_dict["actions"].shape[-1]
        return o + a + 1 + o  # concat(s, a, c_raw, s')

    def sample_starts(self, num, m, np_random, cost_anchor_frac=0.5, cost_goal=None):
        """Start indices of `num` length-`m` windows, a `cost_anchor_frac` fraction of them
        containing a non-zero cost (under `cost_goal`'s cost function, default the recorded one)."""
        alls, anchored = self._valid_starts(m, cost_goal)
        n_anch = int(round(num * cost_anchor_frac)) if len(anchored) else 0
        n_anch = min(n_anch, num)
        picks = []
        if n_anch:
            picks.append(np_random.choice(anchored, size=n_anch, replace=True))
        if num - n_anch:
            picks.append(np_random.choice(alls, size=num - n_anch, replace=True))
        starts = np.concatenate(picks)
        np_random.shuffle(starts)
        return starts

    def segments_at(self, starts, m, cost_goals=None):
        """The windows at `starts` as concat(s, a, c, s') segments.  `cost_goals` (an id, or one id
        per window) swaps the cost column for those regimes' cost bits on the same states."""
        win = starts[:, None] + np.arange(m)[None, :]
        obs = self.dataset_dict["observations"][win]
        act = self.dataset_dict["actions"][win]
        nobs = self.dataset_dict["next_observations"][win]
        if cost_goals is None:
            craw = self.dataset_dict["cost_raw"][win]
        else:
            goals = np.broadcast_to(np.asarray(cost_goals), starts.shape)
            craw = np.empty(win.shape, dtype=np.float32)
            for g in np.unique(goals):
                sel = goals == g
                craw[sel] = self._cost_signal(g)[win[sel]]
        return np.concatenate([obs, act, craw[..., None], nobs], axis=-1).astype(np.float32)

    def sample_segments(self, num, m, np_random, cost_anchor_frac=0.5, return_transitions=False):
        """`num` contiguous length-`m` segments (per step: concat(s, a, c_raw, s')).
        A `cost_anchor_frac` fraction come from windows containing a non-zero cost
        (Sec. `sec:pooled-practical`, Stage 1).  With `return_transitions`, also
        returns the last transition of every segment as a batch dict."""
        starts = self.sample_starts(num, m, np_random, cost_anchor_frac)
        seg = self.segments_at(starts, m)

        if not return_transitions:
            return seg
        last = starts + (m - 1)
        tr = {
            k: self.dataset_dict[k][last]
            for k in ("observations", "actions", "next_observations", "rewards", "costs", "cost_raw", "masks")
        }
        return seg, tr


class PooledSource:
    """The K source regimes (one per goal id in `goal_ids`), plus the samplers the
    six pooled-training stages need."""

    def __init__(self, env_name, goal_ids, reference_path, cost_scale=1.0, seed=0):
        self.np_random = np.random.default_rng(seed)
        self.cost_scale = cost_scale
        self.env_name = env_name
        self.goal_ids = list(goal_ids)
        self.paths = [ant_goal_dataset_path(env_name, g, reference_path) for g in self.goal_ids]
        self.regimes = [
            RegimeDataset(make_multitask_ant_env(env_name, g), regime_id=k, goal_id=g,
                          data_location=self.paths[k], cost_scale=cost_scale)
            for k, g in enumerate(self.goal_ids)
        ]
        for r in self.regimes:
            r.seed(int(self.np_random.integers(1 << 30)))
        self.K = len(self.regimes)
        self.relabeled = False
        print(f"[PooledSource] {self.K} regimes (goals {self.goal_ids}) from {self.paths}")

    def segment_feature_dim(self):
        return self.regimes[0].segment_feature_dim()

    def attach_relabelers(self, costa_dir, device="cpu"):
        """Load COSTA's per-goal cost discriminators and relabel every regime with every other
        regime's cost function (models/data/costa_discriminator.py), enabling
        `sample_contrastive(counterfactual=True)`."""
        from models.data.costa_discriminator import load_discriminators, relabel_regime
        assert self.K > 1, "counterfactual relabeling needs at least two regimes"
        discs, paths = load_discriminators(costa_dir, self.env_name, self.goal_ids, device)
        for r in self.regimes:
            relabel_regime(r, discs, paths, device)
        del discs
        self.relabeled = True

    # -------------------------------------------------- Stage 1: contrastive
    def sample_contrastive(self, per_regime, m, cost_anchor_frac=0.5, counterfactual=False):
        """(segments, regime labels), plus with `counterfactual` the two relabeled batches of COSTA's
        `OfflineMetaLearner` (`create_positive_cost` / `create_negative_cost`), row-aligned with the
        labels:
          pos  regime k's cost function on ANOTHER regime's states -> should embed like regime k
          neg  regime k's own states with ANOTHER regime's cost bits -> should embed away from k
        so the encoder can only separate them by reading the cost column."""
        segs, labels, pos, neg = [], [], [], []
        for k, r in enumerate(self.regimes):
            starts = r.sample_starts(per_regime, m, self.np_random, cost_anchor_frac)
            segs.append(r.segments_at(starts, m))
            labels.append(np.full(per_regime, k, dtype=np.int64))
            if not counterfactual:
                continue
            others = [j for j in range(self.K) if j != k]
            other_goals = np.asarray([self.goal_ids[j] for j in others])
            neg.append(r.segments_at(starts, m, cost_goals=self.np_random.choice(other_goals, size=per_regime)))
            src = self.np_random.choice(others, size=per_regime)
            p = np.empty_like(segs[-1])
            for j in others:
                sel = src == j
                if sel.any():
                    rj = self.regimes[j]
                    pj = rj.sample_starts(int(sel.sum()), m, self.np_random, cost_anchor_frac, cost_goal=r.goal_id)
                    p[sel] = rj.segments_at(pj, m, cost_goals=r.goal_id)
            pos.append(p)
        segs, labels = np.concatenate(segs, axis=0), np.concatenate(labels, axis=0)
        if not counterfactual:
            return segs, labels
        return segs, labels, np.concatenate(pos, axis=0), np.concatenate(neg, axis=0)

    def sample_regime_segments(self, k, num, m, cost_anchor_frac=0.5):
        return self.regimes[k].sample_segments(num, m, self.np_random, cost_anchor_frac)

    # ----------------------------------------------- Stage 2/4: per-regime jax
    def sample_regime_jax(self, k, batch_size):
        return self.regimes[k].sample_jax(batch_size)

    # ------------------------------------ Stage 3: pooled transitions + segments
    def sample_pooled_transitions(self, batch_size, m, cost_anchor_frac=0.5):
        """Returns (segments, transition_batch, regime_labels): `batch_size`
        transitions spread across regimes, each at the end of the length-`m`
        segment used to infer its context (Sec. `sec:pooled-algo` step 3)."""
        per = max(1, batch_size // self.K)
        seg_list, tr_list, lab_list = [], [], []
        for k, r in enumerate(self.regimes):
            seg, tr = r.sample_segments(per, m, self.np_random, cost_anchor_frac, return_transitions=True)
            seg_list.append(seg)
            tr_list.append(tr)
            lab_list.append(np.full(per, k, dtype=np.int64))
        segs = np.concatenate(seg_list, axis=0)
        batch = {k: np.concatenate([t[k] for t in tr_list], axis=0) for k in tr_list[0]}
        labels = np.concatenate(lab_list, axis=0)
        return segs, batch, labels
