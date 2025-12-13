import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class G1Env(gym.Env):
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        # Define paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        xml_path = os.path.join(base_dir, "assets", "humanoid", "humanoid.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find robot XML at {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Define spaces
        # Action space: control all actuators
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.model.nu,), dtype=np.float32)
        
        # Observation space: qpos + qvel (basic)
        obs_shape = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        # Apply action
        # Note: You might want to scale action to actuator limits here
        self.data.ctrl[:] = action 
        
        mujoco.mj_step(self.model, self.data)
        
        # Calculate reward (placeholder: maximize z-height of root)
        # qpos[2] is typically z-height for a free joint root
        reward = self.data.qpos[2] 
        
        terminated = False
        truncated = False
        info = {}
        
        if self.render_mode == "human":
            # In a real training loop, you'd handle rendering differently or use a wrapper
            pass
            
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs(), {}

def main():
    # Test the environment
    env = G1Env()
    obs, _ = env.reset()
    print("Environment created. Observation shape:", obs.shape)
    
    print("Running random actions for 100 steps...")
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            
    print("Test passed!")

if __name__ == "__main__":
    main()
