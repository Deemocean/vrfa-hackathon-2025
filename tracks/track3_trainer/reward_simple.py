"""
Simple reward function for HumanoidStandup task.

Reward components (based on stable standing research):
1. Head height - linear reward capped at target
2. Control cost - torque penalty
3. Falling penalty - penalize negative head velocity
4. Orientation reward - upright torso (roll/pitch near zero)
5. Angular velocity penalty - penalize wobbling/spinning
6. Base acceleration penalty - penalize jerky motions
7. Butt height stage reward - encourage lifting pelvis off ground
8. PD reward - proportional-derivative reward for tracking height targets

References:
- OpenAI Gym HumanoidStandup: uph_cost - quad_ctrl_cost - quad_impact_cost
- Stable Gym: healthy_z_range, health_penalty
- "Revisiting Reward Design for Robust Humanoid Standing and Walking" (2024)
"""

import numpy as np
import mujoco


class RewardCalculator:
    """
    Reward calculator for stable humanoid standup.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Cache body IDs
        self._head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        # Target head height (standing)
        self._target_head_height = 1.5

        # Healthy height range (for standing)
        self._healthy_z_min = 1.0
        self._healthy_z_max = 2.0

        # Target pelvis height for standing (butt should be around hip height)
        self._target_pelvis_height = 0.9

        # PD gains for reward shaping
        self._kp_head = 5.0      # Proportional gain for head height
        self._kd_head = 2.0      # Derivative gain for head height
        self._kp_pelvis = 3.0    # Proportional gain for pelvis height
        self._kd_pelvis = 1.5    # Derivative gain for pelvis height

    def _get_head_height(self):
        """Get height of the head."""
        return self.data.xpos[self._head_id, 2]

    def _get_pelvis_height(self):
        """Get height of the pelvis (butt)."""
        return self.data.xpos[self._pelvis_id, 2]

    def _get_torso_orientation(self):
        """Get torso orientation as rotation matrix."""
        return self.data.xmat[self._torso_id].reshape(3, 3)

    def _get_torso_up_vector(self):
        """Get the 'up' vector of the torso (should point up when standing)."""
        rot_mat = self._get_torso_orientation()
        return rot_mat[:, 2]  # z-axis of torso in world frame

    def compute_reward(self, action, torques):
        """
        Compute reward with stability terms.

        Returns:
            tuple: (total_reward, info_dict)
        """
        head_height = self._get_head_height()

        # === 1. HEAD HEIGHT REWARD ===
        # Stronger incentive to get up - quadratic bonus for higher positions
        height_reward = np.clip(head_height, 0.0, self._target_head_height) * 10.0
        # Bonus for getting above sitting height (encourages standing over sitting)
        if head_height > 0.8:
            height_reward += (head_height - 0.8) * 20.0  # Extra reward above sitting

        # === 2. CONTROL COST ===
        ctrl_cost = -1e-4 * np.sum(torques ** 2)

        # === 3. UPWARD VELOCITY BONUS (encourage getting up) ===
        head_vel_z = self.data.cvel[self._head_id, 5]
        # Reward positive velocity when below target, penalize falling
        if head_height < self._target_head_height:
            velocity_reward = 0.5 * max(0.0, head_vel_z)  # Bonus for moving up
        else:
            velocity_reward = 0.0
        falling_penalty = min(0.0, head_vel_z)  # Always penalize falling

        # === 4. ORIENTATION REWARD (upright torso) ===
        torso_up = self._get_torso_up_vector()
        orientation_score = torso_up[2]  # z component, 1.0 when perfectly upright
        orientation_reward = np.exp(-3.0 * (1.0 - orientation_score) ** 2)

        # === 5. BUTT HEIGHT STAGE REWARD (encourage lifting pelvis) ===
        pelvis_height = self._get_pelvis_height()
        # Progressive reward for lifting butt off ground
        # Stage 1: Any lift above ground level (~0.1m) gets base reward
        # Stage 2: Getting to crouch height (~0.4m) gets more reward
        # Stage 3: Getting to standing height (~0.9m) gets full reward
        butt_stage_reward = 0.0
        if pelvis_height > 0.1:
            butt_stage_reward += 2.0  # Stage 1: off the ground
        if pelvis_height > 0.4:
            butt_stage_reward += 3.0  # Stage 2: crouching
        if pelvis_height > 0.7:
            butt_stage_reward += 5.0  # Stage 3: near standing
        # Smooth continuous component
        butt_stage_reward += np.clip(pelvis_height, 0.0, self._target_pelvis_height) * 5.0

        # === 6. PD REWARD (proportional-derivative for height targets) ===
        # Head PD reward
        head_error = self._target_head_height - head_height
        head_vel_z = self.data.cvel[self._head_id, 5]  # Already computed above
        # P: reward for being close to target (negative error squared)
        # D: reward for moving toward target (positive vel when below target)
        head_pd_reward = -self._kp_head * (head_error ** 2) + self._kd_head * (-head_error * head_vel_z)
        # Clamp D term to avoid rewarding overshooting
        head_pd_reward = np.clip(head_pd_reward, -10.0, 10.0)

        # Pelvis PD reward
        pelvis_error = self._target_pelvis_height - pelvis_height
        pelvis_vel_z = self.data.cvel[self._pelvis_id, 5]
        pelvis_pd_reward = -self._kp_pelvis * (pelvis_error ** 2) + self._kd_pelvis * (-pelvis_error * pelvis_vel_z)
        pelvis_pd_reward = np.clip(pelvis_pd_reward, -10.0, 10.0)

        pd_reward = head_pd_reward + pelvis_pd_reward

        # === 7-9. STABILITY PENALTIES (only apply when standing) ===
        # Don't penalize dynamic motion needed to stand up
        stability_scale = np.clip((head_height - 1.0) / 0.5, 0.0, 1.0)  # 0 below 1m, 1 above 1.5m

        # Angular velocity penalty
        angular_vel = self.data.qvel[3:6]
        angular_vel_magnitude = np.linalg.norm(angular_vel)
        angular_vel_penalty = -0.1 * angular_vel_magnitude * stability_scale

        # Acceleration penalty
        base_acc = self.data.qacc[0:6]
        acc_magnitude = np.linalg.norm(base_acc)
        acc_penalty = -0.01 * acc_magnitude * stability_scale

        # Impact cost
        impact_cost = -0.5e-6 * np.sum(np.square(self.data.cfrc_ext))
        impact_cost = max(impact_cost, -10.0) * stability_scale

        # Total reward
        reward = (
            height_reward +
            ctrl_cost +
            velocity_reward +
            falling_penalty +
            orientation_reward +
            butt_stage_reward +
            pd_reward +
            angular_vel_penalty +
            acc_penalty +
            impact_cost
        )

        # Info dict
        info = {
            "head_height": head_height,
            "head_vel_z": head_vel_z,
            "pelvis_height": pelvis_height,
            "height_reward": height_reward,
            "velocity_reward": velocity_reward,
            "ctrl_cost": ctrl_cost,
            "falling_penalty": falling_penalty,
            "orientation_reward": orientation_reward,
            "orientation_score": orientation_score,
            "butt_stage_reward": butt_stage_reward,
            "pd_reward": pd_reward,
            "head_pd_reward": head_pd_reward,
            "pelvis_pd_reward": pelvis_pd_reward,
            "stability_scale": stability_scale,
            "angular_vel_penalty": angular_vel_penalty,
            "acc_penalty": acc_penalty,
            "impact_cost": impact_cost,
        }

        return reward, info

    def reset(self):
        """Reset calculator state (no state to reset in simple version)."""
        pass
