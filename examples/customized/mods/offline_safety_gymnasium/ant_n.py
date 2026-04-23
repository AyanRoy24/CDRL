# How to use:

# param_dict = {
#     'torso mass': [0.8, 1.2],
#     'leg 1 mass': [0.8, 1.2],
#     # ... other params ...
# }
# env = gym.make('Ant-v3')
# wrapped_env = AntMod(env, param_dict=param_dict)

# Pass a param_dict to the wrapper to enable parameter randomization and scaling, e.g.:
# On every reset, the parameters will be resampled and set.


import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from safety_gymnasium.tasks.safety_velocity.safety_ant_velocity_v0 import SafetyAntVelocityEnv
from gymnasium import spaces
import random

# class AntMod(gym.Wrapper):
# class AntMod(AntEnv):
class AntMod(SafetyAntVelocityEnv):
    # def __init__(self, env, goal_idx=0, max_episode_steps=300, gravity=-4.905, param_dict=None, **kwargs):
    def __init__(self, goal_idx=0, max_episode_steps=300, gravity=-4.905, param_dict=None, **kwargs):
        # super().__init__(env)
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )
        self.goal_idx = goal_idx
        self._step = 0
        self._max_episode_steps = max_episode_steps
        self.new_gravity = gravity

        self.param_dict = param_dict if param_dict is not None else dict()
        self.params = list(self.param_dict.keys())
        self.initial_param_dict = {param: [] for param in self.params}
        self.current_param_scale = dict()

        # Store initial values for all parameters
        model = self.model
        for param in self.params:
            if param == 'torso mass':
                self.initial_param_dict[param].append(model.body_mass[1])
            elif param == 'leg 1 mass':
                self.initial_param_dict[param].append(model.body_mass[3])
            elif param == 'leg 2 mass':
                self.initial_param_dict[param].append(model.body_mass[6])
            elif param == 'leg 3 mass':
                self.initial_param_dict[param].append(model.body_mass[9])
            elif param == 'leg 4 mass':
                self.initial_param_dict[param].append(model.body_mass[12])
            elif param == 'leg mass':
                self.initial_param_dict[param].extend([model.body_mass[3], model.body_mass[6], model.body_mass[9], model.body_mass[12]])
            elif param == 'shin 1 mass':
                self.initial_param_dict[param].append(model.body_mass[2])
            elif param == 'shin 2 mass':
                self.initial_param_dict[param].append(model.body_mass[5])
            elif param == 'shin 3 mass':
                self.initial_param_dict[param].append(model.body_mass[8])
            elif param == 'shin 4 mass':
                self.initial_param_dict[param].append(model.body_mass[11])
            elif param == 'shin mass':
                self.initial_param_dict[param].extend([model.body_mass[2], model.body_mass[5], model.body_mass[8], model.body_mass[11]])
            elif param == 'ankle 1 fric':
                self.initial_param_dict[param].append(model.geom_friction[4][0])
            elif param == 'ankle 2 fric':
                self.initial_param_dict[param].append(model.geom_friction[7][0])
            elif param == 'ankle 3 fric':
                self.initial_param_dict[param].append(model.geom_friction[10][0])
            elif param == 'ankle 4 fric':
                self.initial_param_dict[param].append(model.geom_friction[13][0])
            elif param == 'leg 1 fric':
                self.initial_param_dict[param].append(model.geom_friction[3][0])
            elif param == 'leg 2 fric':
                self.initial_param_dict[param].append(model.geom_friction[6][0])
            elif param == 'leg 3 fric':
                self.initial_param_dict[param].append(model.geom_friction[9][0])
            elif param == 'leg 4 fric':
                self.initial_param_dict[param].append(model.geom_friction[12][0])
            elif param == 'hip damping':
                self.initial_param_dict[param].extend([model.dof_damping[6], model.dof_damping[8], model.dof_damping[10], model.dof_damping[12]])
            elif param == 'ankle damping':
                self.initial_param_dict[param].extend([model.dof_damping[7], model.dof_damping[9], model.dof_damping[11], model.dof_damping[13]])
            elif param == 'fl hip lower limit':
                self.initial_param_dict[param].append(model.jnt_range[1][0])
            elif param == 'fl hip upper limit':
                self.initial_param_dict[param].append(model.jnt_range[1][1])
            elif param == 'fl ankle lower limit':
                self.initial_param_dict[param].append(model.jnt_range[2][0])
            elif param == 'fl ankle upper limit':
                self.initial_param_dict[param].append(model.jnt_range[2][1])
            elif param == 'fr hip lower limit':
                self.initial_param_dict[param].append(model.jnt_range[3][0])
            elif param == 'fr hip upper limit':
                self.initial_param_dict[param].append(model.jnt_range[3][1])
            elif param == 'fr ankle lower limit':
                self.initial_param_dict[param].append(model.jnt_range[4][0])
            elif param == 'fr ankle upper limit':
                self.initial_param_dict[param].append(model.jnt_range[4][1])
            else:
                raise NotImplementedError(f"{param} is not adjustable in Ant")
            self.current_param_scale[param] = 1

        self.apply_modifications()

    def apply_modifications(self):
        model = self.model
        model.opt.gravity[2] = self.new_gravity
        print(f"Dynamics Modified: Gravity set to {model.opt.gravity}")

    def set_params(self, param_scales):
        model = self.model
        assert len(param_scales) == len(self.params), 'Length of new params must align the initialization params'
        for param, scale in param_scales.items():
            if param == 'torso mass':
                model.body_mass[1] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 1 mass':
                model.body_mass[3] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 2 mass':
                model.body_mass[6] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 3 mass':
                model.body_mass[9] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 4 mass':
                model.body_mass[12] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg mass':
                model.body_mass[3] = self.initial_param_dict[param][0] * scale
                model.body_mass[6] = self.initial_param_dict[param][1] * scale
                model.body_mass[9] = self.initial_param_dict[param][2] * scale
                model.body_mass[12] = self.initial_param_dict[param][3] * scale
            elif param == 'shin 1 mass':
                model.body_mass[2] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin 2 mass':
                model.body_mass[5] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin 3 mass':
                model.body_mass[8] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin 4 mass':
                model.body_mass[11] = self.initial_param_dict[param][-1] * scale
            elif param == 'shin mass':
                model.body_mass[2] = self.initial_param_dict[param][0] * scale
                model.body_mass[5] = self.initial_param_dict[param][1] * scale
                model.body_mass[8] = self.initial_param_dict[param][2] * scale
                model.body_mass[11] = self.initial_param_dict[param][3] * scale
            elif param == 'ankle 1 fric':
                model.geom_friction[4][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'ankle 2 fric':
                model.geom_friction[7][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'ankle 3 fric':
                model.geom_friction[10][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'ankle 4 fric':
                model.geom_friction[13][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 1 fric':
                model.geom_friction[3][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 2 fric':
                model.geom_friction[6][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 3 fric':
                model.geom_friction[9][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'leg 4 fric':
                model.geom_friction[12][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'hip damping':
                model.dof_damping[6] = self.initial_param_dict[param][0] * scale
                model.dof_damping[8] = self.initial_param_dict[param][1] * scale
                model.dof_damping[10] = self.initial_param_dict[param][2] * scale
                model.dof_damping[12] = self.initial_param_dict[param][3] * scale
            elif param == 'ankle damping':
                model.dof_damping[7] = self.initial_param_dict[param][0] * scale
                model.dof_damping[9] = self.initial_param_dict[param][1] * scale
                model.dof_damping[11] = self.initial_param_dict[param][2] * scale
                model.dof_damping[13] = self.initial_param_dict[param][3] * scale
            elif param == 'fl hip lower limit':
                model.jnt_range[1][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'fl hip upper limit':
                model.jnt_range[1][1] = self.initial_param_dict[param][-1] * scale
            elif param == 'fl ankle lower limit':
                model.jnt_range[2][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'fl ankle upper limit':
                model.jnt_range[2][1] = self.initial_param_dict[param][-1] * scale
            elif param == 'fr hip lower limit':
                model.jnt_range[3][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'fr hip upper limit':
                model.jnt_range[3][1] = self.initial_param_dict[param][-1] * scale
            elif param == 'fr ankle lower limit':
                model.jnt_range[4][0] = self.initial_param_dict[param][-1] * scale
            elif param == 'fr ankle upper limit':
                model.jnt_range[4][1] = self.initial_param_dict[param][-1] * scale
            else:
                raise NotImplementedError(f"{param} is not adjustable in Ant")
            self.current_param_scale[param] = scale

    def resample_params(self):
        new_scales = {}
        for param, bound_or_possible_values in self.param_dict.items():
            if len(bound_or_possible_values) == 2:
                new_scales[param] = random.uniform(
                    bound_or_possible_values[0],
                    bound_or_possible_values[1]
                )
            else:
                new_scales[param] = random.choice(bound_or_possible_values)
        self.set_params(new_scales)

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))
        if self.goal_idx in [0, 1, 2]:
            direct = (np.cos(0.), np.sin(0.))
        elif self.goal_idx in [3, 4]:
            direct = (np.cos(-np.pi/6), np.sin(-np.pi/6))
        else:
            raise ValueError("Invalid goal_idx")
        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
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
        ctrl_cost = 0.5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        done_cost = float(done)
        cost = np.clip(obj_cost + done_cost, 0, 1)
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
        x = self.data.qpos.flat[0]
        y = self.data.qpos.flat[1]
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            [x / 5.0],
            [y],
        ])

    def reset(self, **kwargs):
        self._step = 0
        # Resample parameters if param_dict is provided
        if self.param_dict:
            self.resample_params()
        # obs, info = self.env.reset(**kwargs)
        obs, info = super().reset(**kwargs)
        self.apply_modifications()
        return obs, info

    @property
    def current_param_scales(self):
        return self.current_param_scale

    @property
    def current_flat_scale(self):
        return list(self.current_param_scale.values())