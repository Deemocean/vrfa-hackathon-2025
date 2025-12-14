"""
PPO Training for HumanoidStandup Task

Techniques to escape sitting local minima (from SOTA research):
1. Assistive force curriculum - upward force that decays over training
2. Multi-stage rewards based on height thresholds
3. Progressive constraint relaxation

References:
- HoST: Learning Humanoid Standing-up Control across Diverse Postures (2025)
- Two-Stage RL Framework for Humanoid Sitting and Standing-Up (PMC, 2024)
"""

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import torch
import mujoco
import mujoco.viewer
import numpy as np
import os

from host_rewards import HoSTRewardCalculator as RewardCalculator


class G1Env(gym.Env):
    """Custom Gymnasium environment for humanoid standup task using MuJoCo."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, assistive_force=200.0):
        self.render_mode = render_mode

        # Define paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        xml_path = os.path.join(base_dir, "assets", "humanoid", "humanoid.xml")

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find robot XML at {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Viewer for human rendering
        self.viewer = None

        # Episode parameters (from paper: 2000 timesteps per episode)
        self.max_episode_steps = 800
        self.current_step = 0

        # Define spaces
        # Action space: control all actuators (scaled to [-1, 1])
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.model.nu,), dtype=np.float32)

        # Observation space: qpos + qvel
        obs_shape = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        # Get torso body id for assistive force
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        # Assistive force curriculum (from HoST paper)
        # Starts high to help robot experience standing, decays over training
        self._assistive_force = assistive_force  # Initial upward force (N)
        self._assistive_force_decay = 0.9999  # Decay per step
        self._min_assistive_force = 0.0  # Final force

        # HoST-style curriculum parameters
        self._curriculum_force_reduction = 20.0  # Force reduction when threshold met
        self._curriculum_height_threshold = 0.9  # Height to trigger force reduction
        self._height_history = []  # Track recent heights for curriculum
        self._height_history_len = 100  # Window for averaging

        # HoST reward calculator
        self.reward_calc = RewardCalculator(self.model, self.data)

        # Observation history for temporal features (HoST uses 6 frames)
        self._obs_history_len = 6
        self._obs_history = []


    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _get_base_height(self):
        """Get the height of the base (torso) body."""
        return self.data.xpos[self._torso_id, 2]

    def step(self, action):
        # Scale action to actuator control range
        ctrl_range = self.model.actuator_ctrlrange
        scaled_action = ctrl_range[:, 0] + (action + 1) * 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        self.data.ctrl[:] = scaled_action

        # Apply assistive upward force on torso (curriculum learning)
        # This helps the robot experience what standing feels like early in training
        # Force is applied only when robot is trying to get up (not already standing)
        base_height = self._get_base_height()
        if self._assistive_force > self._min_assistive_force and base_height < 1.0:
            # Apply upward force to torso body
            self.data.xfrc_applied[self._torso_id, 2] = self._assistive_force
        else:
            self.data.xfrc_applied[self._torso_id, 2] = 0.0

        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        # HoST-style curriculum: reduce force when robot achieves height
        self._height_history.append(base_height)
        if len(self._height_history) > self._height_history_len:
            self._height_history.pop(0)

        # If mean height exceeds threshold, reduce assistive force (HoST curriculum)
        if len(self._height_history) >= self._height_history_len:
            mean_height = np.mean(self._height_history)
            if mean_height > self._curriculum_height_threshold:
                self._assistive_force = max(
                    self._min_assistive_force,
                    self._assistive_force - self._curriculum_force_reduction
                )
                self._height_history = []  # Reset history after reduction

        # Also decay assistive force over time (backup curriculum)
        self._assistive_force = max(
            self._min_assistive_force,
            self._assistive_force * self._assistive_force_decay
        )

        # === REWARD FUNCTION  ===
        torques = self.data.ctrl
        reward, info = self.reward_calc.compute_reward(action, torques)

        # Add assistive force info
        info["assistive_force"] = self._assistive_force

        # Termination conditions
        terminated = False

        # Truncation (episode time limit)
        truncated = self.current_step >= self.max_episode_steps

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Load the "supine" keyframe (lying face up) for standup task
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "supine")
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)

        # Add small random noise to initial state (uniform distribution centered on reference)
        if self.np_random is not None:
            noise_scale = 0.01
            qpos_noise = self.np_random.uniform(-noise_scale, noise_scale, size=self.model.nq)
            qvel_noise = self.np_random.uniform(-noise_scale, noise_scale, size=self.model.nv)
            # Don't add noise to quaternion (indices 3-6)
            qpos_noise[3:7] = 0
            self.data.qpos[:] += qpos_noise
            self.data.qvel[:] += qvel_noise

        # Forward kinematics to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        # Reset reward calculator state
        self.reward_calc.reset()

        # Reset observation history
        self._obs_history = []

        return self._get_obs(), {}

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def make_env(render_mode=None):
    """Factory function to create G1Env instances."""
    def _init():
        return G1Env(render_mode=render_mode)
    return _init


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints during training."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "model", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_latest")
            self.model.save(model_path)

            # Save VecNormalize statistics
            if isinstance(self.training_env, VecNormalize):
                vecnorm_path = os.path.join(self.save_path, "vecnormalize_latest.pkl")
                self.training_env.save(vecnorm_path)

            if self.verbose > 0:
                print(f"Checkpoint saved at step {self.num_timesteps}")
        return True


def train(
    total_timesteps: int = 10_000_000,
    n_envs: int = 8,
    log_dir: str = "./logs/humanoid_standup/",
    resume: bool = False,
):
    """
    Train PPO on HumanoidStandup task.

    Based on paper settings:
    - 200000 timesteps per batch for bipedal standing
    - 2000 timesteps per episode
    - timestep = 0.01 seconds

    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        log_dir: Directory for logs and checkpoints
        resume: Whether to resume from latest checkpoint
    """
    os.makedirs(log_dir, exist_ok=True)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    vecnorm_path = os.path.join(log_dir, "vecnormalize_latest.pkl")

    # Create vectorized environment with normalization
    env = make_vec_env(
        make_env(),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    # Wrap with VecNormalize for observation and reward normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Create evaluation environment
    eval_env = make_vec_env(make_env(), n_envs=1)
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # Policy architecture
    policy_kwargs = dict(
        net_arch=[512, 256, 128],
        activation_fn=torch.nn.ReLU,
    )

    # Load existing model or create new one
    existing_model = None
    if resume:
        candidates = [
            os.path.join(log_dir, "ppo_humanoid_standup_latest.zip"),
            os.path.join(log_dir, "best_model.zip"),
            os.path.join(log_dir, "ppo_humanoid_standup_final.zip"),
        ]
        for p in candidates:
            if os.path.exists(p):
                existing_model = p
                break

        # Load VecNormalize statistics BEFORE loading model
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env.venv)
            env.training = True
            print("Loaded VecNormalize statistics")

    if existing_model:
        print(f"Resuming from: {existing_model}")
        model = PPO.load(existing_model, env=env, device=device)
    else:
        print("Creating new model...")
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=4096,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
        )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=50_000 // n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=log_dir,
        name_prefix="ppo_humanoid_standup",
    )

    # Train!
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"Using {n_envs} parallel environments")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name="ppo_humanoid_standup",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        final_model_path = os.path.join(log_dir, "ppo_humanoid_standup_final")
        model.save(final_model_path)
        env.save(os.path.join(log_dir, "vecnormalize_final.pkl"))
        print(f"Final model saved to {final_model_path}")

    env.close()
    eval_env.close()
    print("Training complete!")


def test(model_path: str = None, vecnorm_path: str = None, log_dir: str = "./logs/humanoid_standup/"):
    """
    Test a trained model.

    Args:
        model_path: Path to the trained model
        vecnorm_path: Path to VecNormalize statistics
        log_dir: Directory containing logs/checkpoints
    """
    if model_path is None:
        model_path = os.path.join(log_dir, "best_model.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(log_dir, "ppo_humanoid_standup_final.zip")

    if vecnorm_path is None:
        vecnorm_path = os.path.join(log_dir, "vecnormalize_final.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from: {model_path}")

    def make_test_env():
        # No assistive force during testing - robot must stand on its own
        return G1Env(render_mode="human", assistive_force=0.0)

    env = DummyVecEnv([make_test_env])

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print("Loaded VecNormalize statistics")

    model = PPO.load(model_path)

    obs = env.reset()
    inner_env = env.envs[0] if not isinstance(env, VecNormalize) else env.venv.envs[0]
    print(f"Starting base height: {inner_env._get_base_height():.3f}")
    print("Testing trained model... (Close window to exit)")

    total_reward = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            total_reward += reward[0]

            if isinstance(env, VecNormalize):
                env.venv.envs[0].render()
            else:
                env.envs[0].render()

            import time
            time.sleep(0.02)

            if dones[0]:
                env.close()
                print(f"Total reward: {total_reward:.2f}")
                break
                

    except KeyboardInterrupt:
        print("\nTesting interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test PPO on HumanoidStandup")
    parser.add_argument("mode", choices=["train", "test"], default="train", nargs="?",
                        help="Mode: train or test")
    parser.add_argument("--timesteps", type=int, default=20_000_000,
                        help="Total training timesteps (default: 10M)")
    parser.add_argument("--n-envs", type=int, default=16,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--log-dir", type=str, default="./logs/humanoid_standup/",
                        help="Directory for logs and checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model for testing")

    args = parser.parse_args()

    if args.mode == "test":
        test(model_path=args.model, log_dir=args.log_dir)
    else:
        train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            log_dir=args.log_dir,
            resume=args.resume,
        )
