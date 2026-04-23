import numpy as np
import gymnasium as gym

class AntDirSafe(gym.Wrapper):
    def __init__(self, env, goal_idx=0, velocity_threshold=1.0, gravity=-4.905):
        super().__init__(env)
        self.goal_idx = goal_idx
        self.velocity_threshold = velocity_threshold
        # Define target directions for Goals 0, 1, 2
        # Example: 0=North, 1=South, 2=East
        self.target_angles = [0, np.pi, np.pi/2]
        self.new_gravity = gravity
        # In Gymnasium MuJoCo envs, the model is usually under env.unwrapped.model
        # We set it once at initialization
        self.apply_modifications()
    
    def apply_modifications(self):
        # Access the MuJoCo model
        # For Gymnasium v0.26+: self.env.unwrapped.model
        # For older versions: self.env.unwrapped.sim.model
        model = self.env.unwrapped.model
        
        # model.opt.gravity is a 3-element array [x, y, z]
        # Earth gravity is [0, 0, -9.81]
        model.opt.gravity[2] = self.new_gravity
        
        # You can also modify the timestep here as requested:
        # model.opt.timestep = 0.01 
        
        print(f"Dynamics Modified: Gravity set to {model.opt.gravity}")

    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Extract velocity (indices depend on your specific Ant/Point config)
        # Usually obs[13:15] for Ant or found in info['velocity']
        vx = info.get('x_velocity', 0)
        vy = info.get('y_velocity', 0)
        velocity = np.sqrt(vx**2 + vy**2)
        
        # 2. Calculate Directional Constraint
        target_angle = self.target_angles[self.goal_idx]
        current_angle = np.arctan2(vy, vx)
        angle_diff = np.abs(current_angle - target_angle)
        
        # 3. Define the Cost (Constraint)
        # If speed is high AND moving in the wrong direction, it's a violation
        if velocity > self.velocity_threshold and angle_diff > (np.pi / 4):
            cost = 1.0
        else:
            cost = 0.0
            
        info['cost'] = cost
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Re-apply modifications on reset to ensure they persist
        obs, info = self.env.reset(**kwargs)
        self.apply_modifications()
        return obs, info

# class ModifiedDynamicsWrapper(gym.Wrapper):
#     def __init__(self, env, gravity=-4.905):
#         super().__init__(env)
#         self.new_gravity = gravity
#         # In Gymnasium MuJoCo envs, the model is usually under env.unwrapped.model
#         # We set it once at initialization
#         self.apply_modifications()

#     def apply_modifications(self):
#         # Access the MuJoCo model
#         # For Gymnasium v0.26+: self.env.unwrapped.model
#         # For older versions: self.env.unwrapped.sim.model
#         model = self.env.unwrapped.model
        
#         # model.opt.gravity is a 3-element array [x, y, z]
#         # Earth gravity is [0, 0, -9.81]
#         model.opt.gravity[2] = self.new_gravity
        
#         # You can also modify the timestep here as requested:
#         # model.opt.timestep = 0.01 
        
#         print(f"Dynamics Modified: Gravity set to {model.opt.gravity}")

#     def reset(self, **kwargs):
#         # Re-apply modifications on reset to ensure they persist
#         obs, info = self.env.reset(**kwargs)
#         self.apply_modifications()
#         return obs, info