import numpy as np
import gymnasium as gym

class AntDirSafe(gym.Wrapper):
    def __init__(self, env, goal_idx=0, max_episode_steps=300, gravity=-4.905):
        super().__init__(env)
        self.goal_idx = goal_idx
        self._step = 0
        self._max_episode_steps = max_episode_steps
        self.new_gravity = gravity
        self.apply_modifications()

    def apply_modifications(self):
        model = self.env.unwrapped.model
        model.opt.gravity[2] = self.new_gravity
        print(f"Dynamics Modified: Gravity set to {model.opt.gravity}")

    def step(self, action):
        # Get torso position before simulation
        torso_xyz_before = np.array(self.env.get_body_com("torso"))

        # Set direction vector based on goal
        if self.goal_idx in [0, 1, 2]:
            direct = (np.cos(0.), np.sin(0.))
        elif self.goal_idx in [3, 4]:
            direct = (np.cos(-np.pi/6), np.sin(-np.pi/6))
        else:
            raise ValueError("Invalid goal_idx")

        # Step the environment
        self.env.do_simulation(action, self.env.frame_skip)
        torso_xyz_after = np.array(self.env.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.env.dt), direct)

        xposafter = self.env.get_body_com("torso")[0]
        yposafter = self.env.get_body_com("torso")[1]

        # Cost constraints for each goal
        obj_cost = 0.0
        if self.goal_idx == 0:
            if yposafter > 3 or yposafter < -3:
                obj_cost = 1.0
        elif self.goal_idx == 1:
            if xposafter >= 0 and xposafter <= 10:
                if yposafter > 4 + xposafter or yposafter < -4 + xposafter:
                    obj_cost = 1.0
            elif xposafter > 10 and xposafter <= 30:
                if yposafter > 24 - xposafter or yposafter < 16 - xposafter:
                    obj_cost = 1.0
            elif xposafter > 30 and xposafter <= 40:
                if yposafter > xposafter - 36 or yposafter < xposafter - 44:
                    obj_cost = 1.0
            else:
                if yposafter > 3 or yposafter < -3:
                    obj_cost = 1.0
        elif self.goal_idx == 2:
            if xposafter >= 0 and xposafter <= 10:
                if yposafter > 4 - xposafter or yposafter < -4 - xposafter:
                    obj_cost = 1.0
            elif xposafter > 10 and xposafter <= 30:
                if yposafter > xposafter - 16 or yposafter < xposafter - 24:
                    obj_cost = 1.0
            elif xposafter > 30 and xposafter <= 40:
                if yposafter > 44 - xposafter or yposafter < 36 - xposafter:
                    obj_cost = 1.0
            else:
                if yposafter > 3 or yposafter < -3:
                    obj_cost = 1.0
        elif self.goal_idx == 3:
            if xposafter >= 0 and xposafter <= 20:
                if yposafter > 4 + 0.5 * xposafter or yposafter < -4 + 0.5 * xposafter:
                    obj_cost = 1.0
            elif xposafter > 20 and xposafter <= 40:
                if yposafter > -0.5 * xposafter + 24 or yposafter < -0.5 * xposafter + 16:
                    obj_cost = 1.0
            else:
                if yposafter > 3 or yposafter < -3:
                    obj_cost = 1.0
        else:
            raise ValueError("Invalid goal_idx")

        # Control and contact costs
        ctrl_cost = 0.5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.env.data.cfrc_ext, -1, 1))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        # Done and cost logic
        state = self.env.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        done_cost = float(done)
        cost = np.clip(obj_cost + done_cost, 0, 1)

        # Observation
        ob = self._get_obs()

        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True

        info = dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            xposafter=xposafter,
            yposafter=yposafter,
            torso_velocity=torso_velocity,
            cost_obj=obj_cost,
            cost_done=done_cost,
            cost=cost,
        )
        return ob, reward, done, False, info

    def _get_obs(self):
        # Assumes MuJoCo-like data structure
        x = self.env.data.qpos.flat[0]
        y = self.env.data.qpos.flat[1]
        return np.concatenate([
            self.env.data.qpos.flat[2:],
            self.env.data.qvel.flat,
            [x / 5.0],
            [y],
        ])

    def reset(self, **kwargs):
        self._step = 0
        obs, info = self.env.reset(**kwargs)
        self.apply_modifications()
        return obs, info