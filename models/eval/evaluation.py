import numpy as np
from PIL import Image
from typing import Dict
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import time
from models.data.dsrl_datasets import DSRLDataset
from tqdm.auto import trange  # noqa
import matplotlib.pyplot as plt

# eval_num = 1


def offline_evaluation(agent, dataset, num_samples: int = 1000, alpha: float = 0.1, seed: int = 0):
    """
    Evaluate coverage/validity using:
      - initial action from the offline dataset (recorded action at time t)
      - subsequent action drawn from a random empirical policy, applied at the dataset next state (time t+1)

    Notes:
      - Q_h(s,a) (safe critic) is evaluated for: (s_t, a_offline) and (s_{t+1}, a_random).
      - V_h(s) (safe value) is used for the discrete-time CBF validity check:
          h(s_{t+1}) >= (1-alpha) * h(s_t)
      - For the random policy we sample/shuffle actions from the dataset (empirical random policy).
    """
    # draw a batch from the offline dataset
    batch = dataset.sample_jax(num_samples)
    obs = batch["observations"]            # shape (N, ...)
    actions = batch["actions"]             # shape (N, act_dim)
    next_obs = batch["next_observations"]  # shape (N, ...)

    # V_h(s) for current and next states using barrier_values function
    vh_obs = agent.barrier_values(obs)
    vh_next = agent.barrier_values(next_obs)

    # Q_h(s_t, a_offline)
    qh_off_ens = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, obs, actions)
    qh_off = jnp.max(qh_off_ens, axis=0)

    # Random empirical policy: shuffle recorded actions and treat them as actions at next_obs
    rng = np.random.default_rng(seed)
    idx = rng.permutation(actions.shape[0])
    rand_actions = actions[idx]

    # Q_h(s_{t+1}, a_random)
    qh_rand_next_ens = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, next_obs, rand_actions)
    qh_rand_next = jnp.max(qh_rand_next_ens, axis=0)

    # Convert to numpy for metric computations
    vh_obs_np = np.array(vh_obs)
    vh_next_np = np.array(vh_next)
    qh_off_np = np.array(qh_off)
    qh_rand_next_np = np.array(qh_rand_next)

    # Coverage (action-level)
    coverage_q_initial = float(np.mean(qh_off_np <= 0.0))          # initial recorded actions
    coverage_q_random_next = float(np.mean(qh_rand_next_np <= 0.0))# random actions at next state

    # Action-level means
    mean_q_initial = float(qh_off_np.mean())
    mean_q_random_next = float(qh_rand_next_np.mean())

    # Coverage (state-level, V_h)
    coverage_v = float(np.mean(vh_obs_np <= 0.0))

    # Validity (discrete-time CBF condition) using V_h:
    valid_mask = vh_next_np >= (1.0 - alpha) * vh_obs_np
    validity_v = float(np.mean(valid_mask))

    return {
        # "coverage_q_initial_offline": coverage_q_initial,
        # "coverage_q_random_next": coverage_q_random_next,
        # "mean_q_initial_offline": mean_q_initial,
        # "mean_q_random_next": mean_q_random_next,
        "o_coverage": coverage_v,
        "o_validity": validity_v,
        # "num_samples": int(num_samples),
    }


def plot_cbf_cost_vs_safe_value(agent, dataset, modeldir, num_samples=1000):
    """
    Plots immediate cost (x-axis) vs safe value v_h* (y-axis) for CBF agent and saves the plot.
    """
    batch = dataset.sample_jax(num_samples)
    costs = batch["costs"]
    observations = batch["observations"]

    safe_values = agent.safe_value.apply_fn({"params": agent.safe_value.params}, observations)

    plt.figure(figsize=(7,5))
    plt.scatter(costs, safe_values, alpha=0.5)
    plt.xlabel("Immediate Cost")
    plt.ylabel("Safe Value $v_h^*$")
    plt.title("CBF: Immediate Cost vs Safe Value")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(modeldir, "cbf_cost_vs_safe_value.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def check_coverage(barrier_values, threshold=0.0):
    """
    Fraction of states the CBF certifies as safe.
    Since h(s,a) = -c(s,a) and Q_h ≤ 0 means safe, use <= 0
    """
    return np.mean(barrier_values <= threshold)

def check_valid(barrier_values, next_barrier_values, alpha=0.1):
    """
    Fraction of transitions satisfying the discrete-time CBF condition.
    CBF condition: h(x_{t+1}) - h(x_t) + α*h(x_t) >= 0
    Rearranged: h(x_{t+1}) >= (1-α)*h(x_t)
    """
    one_minus_alpha = 1 - alpha
    # For barrier values <= 0 (safe), condition is automatically satisfied
    # For barrier values > 0 (unsafe), check the CBF condition
    safe_mask = (barrier_values <= 0)
    unsafe_mask = (barrier_values > 0)
    
    # CBF condition for unsafe states
    cbf_condition = next_barrier_values >= one_minus_alpha * barrier_values
    
    # Valid if: (safe) OR (unsafe AND CBF condition satisfied)
    valid = safe_mask | (unsafe_mask & cbf_condition)
    return np.mean(valid)

def _surface_to_pil(frame):
    """Convert pygame.Surface or numpy array to PIL.Image."""
    if isinstance(frame, pygame.Surface):
        arr = pygame.surfarray.array3d(frame)  # shape (w, h, 3)
        arr = np.transpose(arr, (1, 0, 2))     # -> (h, w, 3)
        return Image.fromarray(arr.copy())
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame)
    raise TypeError(f"Unsupported frame type: {type(frame)}")

def evaluate_md(seed, env_id,  eval_num, agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False) -> Dict[str, float]:
    episode_rets, episode_costs, episode_lens = [], [], []
    barriers, next_barriers = [], []
    frames_all = []
    # for _ in trange(num_episodes, desc="Evaluating", leave=False):
    for ep_idx in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset(seed=seed)
        agent = agent.reset_budget(jnp.array(obs), jnp.array(20.0, dtype=jnp.float32))
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0

        # collect frames for this episode
        frames = []

        while True:
            if render:
                frame = env.render(mode="top_down", window=False, screen_size=(700, 300) , screen_record=True)
                # convert immediately to PIL.Image (safe) and store
                try:
                    pil_img = _surface_to_pil(frame)
                except TypeError:
                    # fallback: if env returns something else, try Image.fromarray directly
                    pil_img = Image.fromarray(np.array(frame))

                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(pil_img)
                label_text = f"Episode {ep_idx + 1}/{num_episodes}"
                try:
                    # Try to use a larger font if available
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
                except:
                    # Fallback to default font
                    font = ImageFont.load_default()
                
                # Draw text with background for better visibility
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Position: top-left corner with padding
                x, y = 10, 10
                # Draw background rectangle
                draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill=(0, 0, 0, 180))
                # Draw text
                draw.text((x, y), label_text, fill=(255, 255, 255), font=font)

                frames.append(pil_img)
                # frames.append(frame)
                time.sleep(1e-3)
                
            # action, agent = agent.eval_actions(obs)
            action, agent = agent.eval_actions(jnp.array(obs))
            action = np.array(action)            
            # Get barrier value (safe value function) - raw Q_h values
            # barrier_value = agent.safe_value.apply_fn(
            #     {"params": agent.safe_value.params},
            #     jnp.expand_dims(obs, axis=0)
            # ).item()
            barrier_value = agent.barrier_values(jnp.expand_dims(obs, axis=0)).item()

            barriers.append(barrier_value)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # next_barrier_value = agent.safe_value.apply_fn(
            #     {"params": agent.safe_value.params},
            #     jnp.expand_dims(next_obs, axis=0)
            # ).item()
            next_barrier_value = agent.barrier_values(jnp.expand_dims(next_obs, axis=0)).item()            
            next_barriers.append(next_barrier_value)
            
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            obs = next_obs
            
            if terminated or truncated:
                break
                
        
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

        # save GIF for this episode (if requested and frames were captured)
        # if save_video and len(frames) > 0:
            # try:
        # frames[0].save(f"demo_ep{len(episode_rets)}.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)

        '''
        duration: time each frame is shown, in milliseconds (int) or a list of ints (per-frame). Example: duration=50 → each frame shown 50 ms (≈20 fps). You can pass a list like duration=[50,100,...] to vary per-frame timing.
        loop: how many times the animation repeats; 0 means loop forever, a positive integer sets the number of repeats.
        tip: compute duration for a target framerate as duration_ms = 1000 / fps (e.g., fps=30 → duration≈33). Ensure frames is non-empty before calling save
        '''
        if len(frames) > 0:
            frames_all.extend(frames)
            # try:
            # imgs = [frame for frame in frames]
            # imgs = [Image.fromarray(img) for img in imgs]
            # imgs[0].save(f"demo_ep{eval_num}.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
            # frames[0].save(f"demo_ep{eval_num}.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)
            # except Exception as e:
            #     print("Failed to save GIF:", e)
        
    barriers = jnp.array(barriers)
    next_barriers = jnp.array(next_barriers)
    # eval_num += 1
    validity = check_valid(barriers, next_barriers, alpha=0.9)
    
    coverage = check_coverage(barriers, threshold=0.0)

    # if save_video and len(frames_all) > 0:
    if len(frames_all) > 0:
        try:
            pil_images = []
            for img in frames_all:
                if isinstance(img, Image.Image):
                    pil_images.append(img.convert("RGBA"))
                elif isinstance(img, np.ndarray):
                    pil_images.append(Image.fromarray(img).convert("RGBA"))
                else:
                    pil_images.append(Image.fromarray(np.array(img)).convert("RGBA"))

            # Convert to palette mode suitable for GIF
            paletted = [im.convert("P", palette=Image.ADAPTIVE) for im in pil_images]
            out_name = f"{env_id}_{seed}_{eval_num}.gif"
            paletted[0].save(
                out_name,
                format="GIF",
                save_all=True,
                append_images=paletted[1:],
                duration=50,
                loop=0,
            )
            print(f"Saved concatenated GIF: {out_name}")
        except Exception as e:
            print("Failed to save concatenated GIF:", e)

    return {
        "return": np.mean(episode_rets),
        "cost": np.mean(episode_costs),
        "episode_len": np.mean(episode_lens),
        "coverage": coverage,
        "validity": validity,
    }

def evaluate(
    seed, agent, env: gym.Env, num_episodes: int,
    cost_thresholds=(20, 40, 80), save_video: bool = False, render: bool = False,
) -> Dict[str, float]:
    """Budget-conditioned evaluation.

    For each soft-constraint budget in ``cost_thresholds`` the running budget is
    reset at the start of every episode (``agent.reset_budget``) and decremented
    online inside ``agent.eval_actions``.
    """
    barriers, next_barriers = [], []
    results: Dict[str, float] = {}
    all_rets, all_costs, all_lens, all_nret, all_ncost = [], [], [], [], []

    for threshold in cost_thresholds:
        env.set_target_cost(float(threshold))
        threshold_arr = jnp.array(float(threshold), dtype=jnp.float32)
        episode_rets, episode_costs, episode_lens = [], [], []

        for _ in trange(num_episodes, desc=f"Evaluating C={threshold}", leave=False):
            obs, info = env.reset(seed=seed)
            agent = agent.reset_budget(jnp.array(obs), threshold_arr)
            episode_ret, episode_cost, episode_len = 0.0, 0.0, 0

            while True:
                if render:
                    env.render()
                    time.sleep(1e-3)

                action, agent = agent.eval_actions(jnp.array(obs))
                action = np.array(action)
                barriers.append(agent.barrier_values(jnp.expand_dims(obs, axis=0)).item())

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_barriers.append(agent.barrier_values(jnp.expand_dims(next_obs, axis=0)).item())

                episode_ret += reward
                episode_len += 1
                episode_cost += info["cost"]
                obs = next_obs

                if terminated or truncated:
                    break

            episode_rets.append(episode_ret)
            episode_lens.append(episode_len)
            episode_costs.append(episode_cost)

        mean_ret, mean_cost = float(np.mean(episode_rets)), float(np.mean(episode_costs))
        n_ret, n_cost = env.get_normalized_score(mean_ret, mean_cost)
        results[f"budget_{threshold}/return"] = mean_ret
        results[f"budget_{threshold}/cost"] = mean_cost
        results[f"budget_{threshold}/n_return"] = n_ret
        results[f"budget_{threshold}/n_cost"] = n_cost
        results[f"budget_{threshold}/episode_len"] = float(np.mean(episode_lens))
        all_rets.append(mean_ret)
        all_costs.append(mean_cost)
        all_lens.append(float(np.mean(episode_lens)))
        all_nret.append(n_ret)
        all_ncost.append(n_cost)

    barriers = jnp.array(barriers)
    next_barriers = jnp.array(next_barriers)

    results.update({
        "return": float(np.mean(all_rets)),
        "cost": float(np.mean(all_costs)),
        "n_return": float(np.mean(all_nret)),
        "n_cost": float(np.mean(all_ncost)),
        "episode_len": float(np.mean(all_lens)),
        "coverage": check_coverage(barriers, threshold=0.0),
        "validity": check_valid(barriers, next_barriers, alpha=0.9),
    })
    return results


def evaluate_pooled(
    seed, agent, env: gym.Env, num_episodes: int,
    seg_len: int, context_dim: int,
    cost_thresholds=(20, 40, 80), delta_init=None,
) -> Dict[str, float]:
    """Deployment for Part III (Sec. `sec:pooled-algo`, step 6), budget-conditioned
    like ``evaluate``: every soft-constraint budget in ``cost_thresholds`` is rolled
    out on its own and logged as ``budget_{C}/...``; the headline metrics are the
    averages over budgets.

    Per episode: an `seg_len`-step burn-in under the cost-greedy fallback, then
    *freeze* the inferred constraint context zeta_hat (Prop. `prop:context-switching`(1)),
    reset the tracked budget to the threshold under test (`bcrl_det`'s
    `cost_limits`), and roll out sampling from `IQLPooled`'s dual policy
    (budget-conditioned reward actor / unconditioned safe actor, gated on
    whether the remaining budget still covers `V_C(s,zeta)`), advancing the
    budget by `g(...)` after every step. No hard barrier `Q_h` is used here --
    `IQLPooled` (bcrl_det-style) has none.

    Score reporting matches CDSRL/COSTA/COSTA's evaluation (e.g.
    `policy_trainer/mf_meta_safe_policy_trainer.py::_evaluate`): the headline
    ``cost`` is the *cost rate* -- ``episode_cost / episode_length`` -- which
    cannot exceed 1 since every per-step cost from these envs is itself clipped
    to ``[0, 1]`` (see e.g. `rlkit.envs.ant_dir_safe.AntDirEnv.step`). The raw
    per-episode cost sum (COSTA's own ``episode_cost``, unbounded) is kept
    alongside as ``cost_sum`` for reference.

    ``delta_init`` (legacy) evaluates the single budget it names when
    ``cost_thresholds`` is None."""

    def step_cost(info):
        for k in ("cost", "violation", "cost_raw"):
            if k in info:
                return float(info[k])
        return 0.0

    if cost_thresholds is None:
        cost_thresholds = (float(delta_init),)
    # `_max_episode_steps` lives on the base env; the wrap_gym chain hides
    # underscore attrs, so reach it via .unwrapped.
    max_steps = getattr(getattr(env, "unwrapped", env), "_max_episode_steps", 1000)
    results: Dict[str, float] = {}
    all_rets, all_costs, all_cost_sums, all_lens, all_nret, all_ncost = [], [], [], [], [], []
    all_final_budgets, all_ctx_norms = [], []

    for threshold in cost_thresholds:
        # n_cost is reported relative to the budget the agent actually tracks here.
        if hasattr(env, "set_target_cost"):
            env.set_target_cost(float(threshold))
        ep_rets, ep_cost_sums, ep_lens, final_budgets, ctx_norms = [], [], [], [], []

        for _ in trange(num_episodes, desc=f"Evaluating (pooled) C={threshold}", leave=False):
            obs, _ = env.reset(seed=seed)
            agent = agent.set_context(np.zeros(context_dim, np.float32))
            seg_steps, ret, cost, length, done = [], 0.0, 0.0, 0, False

            while len(seg_steps) < seg_len and not done:
                action, agent = agent.eval_actions(jnp.asarray(obs, jnp.float32))
                nobs, r, term, trunc, info = env.step(np.array(action))
                c = step_cost(info)
                seg_steps.append(np.concatenate([
                    np.asarray(obs, np.float32), np.asarray(action, np.float32),
                    np.array([c], np.float32), np.asarray(nobs, np.float32)]))
                ret += r; cost += c; length += 1
                obs = nobs
                done = bool(term) or bool(trunc) or length >= max_steps

            if len(seg_steps) < seg_len:
                ep_rets.append(ret); ep_cost_sums.append(cost); ep_lens.append(length)
                continue

            seg = np.stack(seg_steps, axis=0)[None]
            ctx_vec = np.asarray(agent.infer_context(jnp.asarray(seg))[0])
            ctx_norms.append(float(np.linalg.norm(ctx_vec)))
            agent = agent.set_context(ctx_vec)
            agent = agent.reset_budget(obs, float(threshold))

            while not done:
                action, agent = agent.eval_actions(jnp.asarray(obs, jnp.float32))
                nobs, r, term, trunc, info = env.step(np.array(action))
                agent = agent.step_budget(jnp.asarray(obs, jnp.float32),
                                          jnp.asarray(action, jnp.float32),
                                          jnp.asarray(nobs, jnp.float32))
                ret += r; cost += step_cost(info); length += 1
                obs = nobs
                done = bool(term) or bool(trunc) or length >= max_steps

            ep_rets.append(ret); ep_cost_sums.append(cost); ep_lens.append(length)
            final_budgets.append(float(agent.delta_hat))

        mean_ret, mean_cost_sum = float(np.mean(ep_rets)), float(np.mean(ep_cost_sums))
        mean_len = float(np.mean(ep_lens))
        mean_cost_rate = float(np.mean([c / max(l, 1) for c, l in zip(ep_cost_sums, ep_lens)]))
        mean_fb = float(np.mean(final_budgets)) if final_budgets else 0.0
        mean_ctx = float(np.mean(ctx_norms)) if ctx_norms else 0.0
        results[f"budget_{threshold}/return"] = mean_ret
        results[f"budget_{threshold}/cost"] = mean_cost_rate
        results[f"budget_{threshold}/cost_sum"] = mean_cost_sum
        results[f"budget_{threshold}/episode_len"] = mean_len
        results[f"budget_{threshold}/final_budget"] = mean_fb
        results[f"budget_{threshold}/ctx_norm"] = mean_ctx
        all_rets.append(mean_ret); all_costs.append(mean_cost_rate); all_cost_sums.append(mean_cost_sum)
        all_lens.append(mean_len)
        all_final_budgets.append(mean_fb); all_ctx_norms.append(mean_ctx)
        if hasattr(env, "get_normalized_score"):
            n_ret, n_cost = env.get_normalized_score(mean_ret, mean_cost_sum)
            results[f"budget_{threshold}/n_return"] = float(n_ret)
            results[f"budget_{threshold}/n_cost"] = float(n_cost)
            all_nret.append(float(n_ret)); all_ncost.append(float(n_cost))

    results.update({
        "return": float(np.mean(all_rets)),
        "cost": float(np.mean(all_costs)),          # cost rate, in [0, 1]
        "cost_sum": float(np.mean(all_cost_sums)),   # raw per-episode cost sum (COSTA's own metric)
        "episode_len": float(np.mean(all_lens)),
        "final_budget": float(np.mean(all_final_budgets)),
        "ctx_norm": float(np.mean(all_ctx_norms)),
    })
    if all_nret:
        results["n_return"] = float(np.mean(all_nret))
        results["n_cost"] = float(np.mean(all_ncost))
    return results


def evaluate_pr(
    agent, env: gym.Env, num_episodes: int, cost_threshold: float = 20.0) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens = [], [], []
    barriers, next_barriers = [], []
    threshold_arr = jnp.array(float(cost_threshold), dtype=jnp.float32)

    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs = env.reset()
        agent = agent.reset_budget(jnp.array(obs), threshold_arr)
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        while True:
            action, agent = agent.eval_actions(jnp.array(obs))
            
            # Get barrier value (safe value function)
            barrier_value = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params},
                jnp.expand_dims(obs, axis=0)
            ).item()
            
            barriers.append(barrier_value)
            
            obs, reward, done, info = env.step(action)
            
            next_barrier_value = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params},
                jnp.expand_dims(obs, axis=0)
            ).item()
            
            next_barriers.append(next_barrier_value)
            
            cost = info["violation"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if done or episode_len == env._max_episode_steps:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    barriers = jnp.array(barriers)
    next_barriers = jnp.array(next_barriers)

    validity = check_valid(barriers, next_barriers, alpha=0.9)
    coverage = check_coverage(barriers, threshold=0.0)

    return {
        "return": np.mean(episode_rets),
        "episode_len": np.mean(episode_lens),
        "cost": np.mean(episode_costs),
        "coverage": coverage,
        "validity": validity,
    }