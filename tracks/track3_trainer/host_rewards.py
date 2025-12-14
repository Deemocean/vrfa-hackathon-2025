"""
Reward functions for HumanoidStandup task based on HoST paper (Table VI).

Reward groups:
(a) Task Reward - high-level task objectives (w^task = 2.5)
(b) Style Reward - style of standing-up motion (w^style = 1)
(c) Regularization Reward - regularization on standing-up motion (w^regu = 0.1)
(d) Post-task Reward - desired behaviors after successful standing up (w^post = 1)

Reference: HoST: Learning Humanoid Standing-up Control across Diverse Postures (2025)
"""

import numpy as np
import mujoco


def f_tol(value, bounds, scale, tolerance):
    """
    Gaussian-style function with saturation bound.
    f_tol(x, [low, high], scale, tol) returns:
    - 1.0 if low <= x <= high
    - exp(-scale * (x - bound)^2) otherwise, clamped by tolerance
    """
    low, high = bounds
    if low <= value <= high:
        return 1.0
    elif value < low:
        return np.exp(-scale * (value - low) ** 2)
    else:
        return np.exp(-scale * (value - high) ** 2)


class HoSTRewardCalculator:
    """
    Calculates rewards based on HoST paper Table VI.
    """

    # Standing stage height thresholds
    H_STAGE1 = 0.5  # Initial rising stage
    H_STAGE2 = 1.0  # Standing stage

    # Group weights - increased task weight to prioritize standing up
    W_TASK = 5.0   # Increased from 2.5 to emphasize getting up
    W_STYLE = 1.1  # Reduced from 1.0 to allow more exploration early on
    W_REGU = 0.1  # Reduced from 0.1 to not over-penalize movement
    W_POST = 2.0

    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Cache body/geom/joint IDs
        self._cache_ids()

        # Previous action for action rate calculation
        self._prev_action = None
        self._prev_prev_action = None

    def _cache_ids(self):
        """Cache MuJoCo object IDs for efficient lookup."""
        # Bodies
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")

        # Foot geoms
        self._foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot1_right"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot2_right"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot1_left"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot2_left"),
        ]
        self._floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Joint indices (for qpos access)
        # Spinal joints
        self._abdomen_z_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "abdomen_z")
        self._abdomen_y_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "abdomen_y")
        self._abdomen_x_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "abdomen_x")

        # Hip joints
        self._hip_y_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_y_right")
        self._hip_y_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_y_left")
        self._hip_x_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_x_right")
        self._hip_x_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_x_left")
        self._hip_z_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_z_right")
        self._hip_z_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_z_left")

        # Knee joints
        self._knee_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_right")
        self._knee_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_left")

        # Ankle joints
        self._ankle_y_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_y_right")
        self._ankle_y_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_y_left")
        self._ankle_x_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_x_right")
        self._ankle_x_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_x_left")

        # Shoulder joints
        self._shoulder1_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder1_right")
        self._shoulder1_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder1_left")
        self._shoulder2_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder2_right")
        self._shoulder2_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder2_left")

        # Shin bodies (for shank orientation)
        self._shin_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "shin_right")
        self._shin_left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "shin_left")

        # Foot bodies
        self._foot_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_right")
        self._foot_left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_left")

    def _get_joint_qpos(self, joint_id):
        """Get joint position from qpos array."""
        # For hinge joints, qpos index = joint_id + 6 (after freejoint's 7 DOF - 1)
        # Actually need to use model.jnt_qposadr
        qpos_adr = self.model.jnt_qposadr[joint_id]
        return self.data.qpos[qpos_adr]

    def _get_joint_qvel(self, joint_id):
        """Get joint velocity from qvel array."""
        qvel_adr = self.model.jnt_dofadr[joint_id]
        return self.data.qvel[qvel_adr]

    def _get_base_height(self):
        """Get height of the base (torso)."""
        return self.data.xpos[self._torso_id, 2]

    def _get_head_height(self):
        """Get height of the head."""
        return self.data.xpos[self._head_id, 2]

    def _get_base_orientation(self):
        """Get base orientation (z-component of projected gravity vector)."""
        # Get torso rotation matrix
        torso_xmat = self.data.xmat[self._torso_id].reshape(3, 3)
        # Gravity vector in world frame is [0, 0, -1]
        # Project to body frame
        gravity_body = torso_xmat.T @ np.array([0, 0, -1])
        # Return z-component (should be close to -1 when upright)
        return -gravity_body[2]  # Negate so upright = +1

    def _get_base_angular_velocity(self):
        """Get base angular velocity (xy components)."""
        # Angular velocity is in qvel[3:6] for freejoint
        return self.data.qvel[3:6]

    def _get_base_linear_velocity(self):
        """Get base linear velocity (xy components)."""
        return self.data.qvel[0:3]

    def _get_feet_contact_count(self):
        """Count foot geoms in contact with floor."""
        count = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 == self._floor_geom_id and g2 in self._foot_geom_ids) or \
               (g2 == self._floor_geom_id and g1 in self._foot_geom_ids):
                count += 1
        return count

    def _get_feet_contact_forces(self):
        """Get contact forces for feet (returns list of (normal, tangential) tuples)."""
        forces = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 == self._floor_geom_id and g2 in self._foot_geom_ids) or \
               (g2 == self._floor_geom_id and g1 in self._foot_geom_ids):
                # Get contact force
                c_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_force)
                normal = c_force[0]  # Normal force
                tangential = np.sqrt(c_force[1]**2 + c_force[2]**2)  # Tangential force magnitude
                forces.append((normal, tangential))
        return forces

    def _get_foot_positions(self):
        """Get foot positions (xy)."""
        right = self.data.xpos[self._foot_right_id, :2]
        left = self.data.xpos[self._foot_left_id, :2]
        return right, left

    def _get_foot_heights(self):
        """Get foot heights."""
        right = self.data.xpos[self._foot_right_id, 2]
        left = self.data.xpos[self._foot_left_id, 2]
        return right, left

    # =========================================================================
    # (a) TASK REWARDS (w^task = 2.5)
    # =========================================================================

    def task_head_height(self):
        """Head height reward: f_tol(h_head, [1, inf], 1, 0.1)"""
        h_head = self._get_head_height()
        return f_tol(h_head, [1.0, np.inf], 1.0, 0.1)

    def task_base_orientation(self):
        """Base orientation reward: f_tol(-θ^z_base, [0.99, inf], 1, 0.05)"""
        orientation = self._get_base_orientation()
        return f_tol(orientation, [0.99, np.inf], 1.0, 0.05)

    def task_progressive_height(self):
        """
        Dense progressive height reward to escape sitting local minima.
        Provides continuous gradient for any upward movement.
        """
        h_base = self._get_base_height()
        h_pelvis = self.data.xpos[self._pelvis_id, 2]

        # Target heights
        target_base = 1.28
        target_pelvis = 0.9

        # Dense rewards - linear scaling with height (always provides gradient)
        base_reward = h_base / target_base  # 0 to 1+ as robot rises
        pelvis_reward = h_pelvis / target_pelvis  # 0 to 1+ as pelvis rises

        # Stage bonuses (sparse but helpful milestones)
        stage_bonus = 0.0
        if h_pelvis > 0.2:  # Got pelvis slightly off ground
            stage_bonus += 1.0
        if h_pelvis > 0.4:  # Transitioning from lying
            stage_bonus += 2.0
        if h_pelvis > 0.6:  # Kneeling/crouching
            stage_bonus += 3.0
        if h_pelvis > 0.8:  # Nearly standing pelvis
            stage_bonus += 5.0
        if h_base > 0.8:  # Torso rising
            stage_bonus += 3.0
        if h_base > 1.0:  # Nearly standing
            stage_bonus += 5.0
        if h_base > 1.1 and h_pelvis > 0.85:  # Standing!
            stage_bonus += 10.0

        return base_reward + pelvis_reward + stage_bonus

    def task_feet_contact(self):
        """
        Reward for feet contact when attempting to stand.
        Penalize losing foot contact when pelvis is high.
        """
        h_pelvis = self.data.xpos[self._pelvis_id, 2]
        feet_count = self._get_feet_contact_count()

        if h_pelvis < 0.4:
            # Still lying down, no feet contact requirement
            return 0.0

        if feet_count > 0:
            # Reward feet on ground (0.5 per foot geom, max ~2.0)
            return 0.5 * feet_count
        else:
            # Penalty for losing all foot contact when should be standing
            return -10.0

    def task_standing_stability(self):
        """
        Reward for maintaining stable standing position.
        Balanced approach: encourage standing, gently discourage jumping.
        """
        h_base = self._get_base_height()
        h_pelvis = self.data.xpos[self._pelvis_id, 2]
        v = self._get_base_linear_velocity()
        v_z = v[2]  # Vertical velocity

        reward = 0.0
        feet_count = self._get_feet_contact_count()

        # === STANDING REWARDS (main driver) ===
        if h_base > 1.0 and h_pelvis > 0.8:
            # Base standing reward
            reward += 10.0

            # Bonus for feet on ground while standing
            if feet_count >= 2:
                reward += 5.0

            # Small bonus for low velocity (stable)
            v_magnitude = np.linalg.norm(v)
            reward += 2.0 * np.exp(-v_magnitude**2)

        # Extra bonus for good standing
        if h_base > 1.15 and h_pelvis > 0.85:
            reward += 5.0

        # === GENTLE ANTI-JUMPING (only when clearly airborne) ===
        # Only penalize if high up AND no feet contact AND moving up
        if h_pelvis > 0.7 and feet_count == 0 and v_z > 1.0:
            reward -= 3.0  # Gentle penalty

        return reward

    def compute_task_reward(self):
        """
        Compute total task reward using MULTIPLICATIVE structure (HoST paper).

        Key insight: By multiplying orientation and height rewards, the robot
        MUST satisfy BOTH constraints simultaneously. Poor balance crashes the episode.
        """
        r_head = self.task_head_height()
        r_orient = self.task_base_orientation()
        r_progressive = self.task_progressive_height()
        r_feet = self.task_feet_contact()
        r_stability = self.task_standing_stability()

        # MULTIPLICATIVE core task rewards (HoST key design)
        # Must satisfy both orientation AND height to get reward
        core_task = r_head * r_orient

        # ADDITIVE auxiliary rewards (help with learning signal)
        auxiliary = r_progressive + r_feet + r_stability

        return core_task + auxiliary

    # =========================================================================
    # (b) STYLE REWARDS (w^style = 1)
    # =========================================================================

    def style_waist_yaw_deviation(self):
        """Waist yaw deviation penalty: -10 if |q_waist| > 1.4"""
        q_waist = self._get_joint_qpos(self._abdomen_z_idx)
        if abs(q_waist) > 1.4:
            return -10.0
        return 0.0

    def style_hip_roll_yaw_deviation(self):
        """Hip roll/yaw deviation penalty: -10 if max hip > 1.4, -10 if min hip > 0.9"""
        # Hip x (roll) and z (yaw) joints
        hip_angles = [
            abs(self._get_joint_qpos(self._hip_x_right_idx)),
            abs(self._get_joint_qpos(self._hip_x_left_idx)),
            abs(self._get_joint_qpos(self._hip_z_right_idx)),
            abs(self._get_joint_qpos(self._hip_z_left_idx)),
        ]
        penalty = 0.0
        if max(hip_angles) > 1.4:
            penalty -= 10.0
        if min(hip_angles) > 0.9:
            penalty -= 10.0
        return penalty

    def style_knee_deviation(self):
        """Knee deviation penalty."""
        knee_right = self._get_joint_qpos(self._knee_right_idx)
        knee_left = self._get_joint_qpos(self._knee_left_idx)
        penalty = 0.0
        # Penalty if max knee angle > 2.85 (weight -0.25 for ground)
        if max(abs(knee_right), abs(knee_left)) > 2.85:
            penalty -= 0.25
        # Penalty if min knee angle < -0.06 (weight -10 for ground)
        if min(knee_right, knee_left) < -0.06:
            penalty -= 10.0
        return penalty

    def style_shoulder_roll_deviation(self):
        """Shoulder roll deviation penalty: -2.5 if shoulder angles out of range"""
        shoulder_left = self._get_joint_qpos(self._shoulder1_left_idx)
        shoulder_right = self._get_joint_qpos(self._shoulder1_right_idx)
        penalty = 0.0
        if max(shoulder_left, shoulder_right) < -0.02:
            penalty -= 2.5
        if min(shoulder_left, shoulder_right) > 0.02:
            penalty -= 2.5
        return penalty

    def style_foot_displacement(self):
        """Foot displacement reward: encourage CoM in support polygon when h > H_stage2."""
        h_base = self._get_base_height()
        if h_base <= self.H_STAGE2:
            return 0.0

        # Get base (CoM) xy position
        base_xy = self.data.xpos[self._torso_id, :2]
        # Get foot positions
        foot_right, foot_left = self._get_foot_positions()
        foot_center = (foot_right + foot_left) / 2

        # Distance from base to foot center
        dist = np.linalg.norm(base_xy - foot_center)
        reward = np.exp(-2 * np.clip(dist, 0, 0.3) ** 2)
        return 2.5 * reward  # Weight 2.5/2.5 for left/right

    def style_ankle_parallel(self):
        """Ankle parallel reward: encourage ankles to be parallel to ground."""
        # Get ankle z-angles (variance should be low)
        ankle_right_y = self._get_joint_qpos(self._ankle_y_right_idx)
        ankle_left_y = self._get_joint_qpos(self._ankle_y_left_idx)
        variance = (np.var([ankle_right_y]) + np.var([ankle_left_y])) / 2
        if variance < 0.05:
            return 20.0
        return 0.0

    def style_foot_distance(self):
        """Foot distance penalty: -10 if feet too far apart (> 0.9m)."""
        foot_right, foot_left = self._get_foot_positions()
        dist = np.linalg.norm(foot_right - foot_left)
        if dist > 0.9:
            return -10.0
        return 0.0

    def style_feet_stumble(self):
        """Feet stumble penalty: penalize horizontal contact force > 3x vertical."""
        forces = self._get_feet_contact_forces()
        for normal, tangential in forces:
            if normal > 0 and tangential > 3 * abs(normal):
                return -10.0  # Using ground weight
        return 0.0

    def style_shank_orientation(self):
        """Shank orientation reward: encourage shanks perpendicular to ground."""
        h_base = self._get_base_height()
        if h_base <= self.H_STAGE1:
            return 0.0

        # Get shank z-angles (should be close to vertical)
        # Using body orientations
        shin_right_z = self.data.xmat[self._shin_right_id].reshape(3, 3)[2, 2]
        shin_left_z = self.data.xmat[self._shin_left_id].reshape(3, 3)[2, 2]
        mean_z = (shin_right_z + shin_left_z) / 2

        reward = f_tol(mean_z, [0.8, np.inf], 1.0, 0.1)
        return 10.0 * reward

    def style_base_angular_velocity(self):
        """Base angular velocity reward: low angular velocity during rising."""
        h_base = self._get_base_height()
        if h_base <= self.H_STAGE1:
            return 0.0

        omega_xy = self._get_base_angular_velocity()[:2]
        reward = np.exp(-2 * np.linalg.norm(omega_xy) ** 2)
        return reward

    def compute_style_reward(self):
        """Compute total style reward."""
        reward = 0.0
        reward += self.style_waist_yaw_deviation()
        reward += self.style_hip_roll_yaw_deviation()
        reward += self.style_knee_deviation()
        reward += self.style_shoulder_roll_deviation()
        reward += self.style_foot_displacement()
        reward += self.style_ankle_parallel()
        reward += self.style_foot_distance()
        reward += self.style_feet_stumble()
        reward += self.style_shank_orientation()
        reward += self.style_base_angular_velocity()
        return reward

    # =========================================================================
    # (c) REGULARIZATION REWARDS (w^regu = 0.1)
    # =========================================================================

    def regu_joint_acceleration(self, qacc):
        """Joint acceleration penalty: -2.5e-7 * ||q̈||²"""
        return -2.5e-7 * np.sum(qacc ** 2)

    def regu_action_rate(self, action):
        """Action rate penalty: -1e-2 * ||a_t - a_{t-1}||²"""
        if self._prev_action is None:
            return 0.0
        diff = action - self._prev_action
        return -1e-2 * np.sum(diff ** 2)

    def regu_smoothness(self, action):
        """Smoothness penalty: -1e-2 * ||a_t - 2*a_{t-1} + a_{t-2}||²"""
        if self._prev_action is None or self._prev_prev_action is None:
            return 0.0
        diff = action - 2 * self._prev_action + self._prev_prev_action
        return -1e-2 * np.sum(diff ** 2)

    def regu_torques(self, torques):
        """Torque penalty: -2.5e-6 * ||τ||²"""
        return -2.5e-6 * np.sum(torques ** 2)

    def regu_joint_power(self, torques, qvel):
        """Joint power penalty: -2.5e-5 * |τ||q̇|"""
        # Use actuator velocities (last nv - 6 for actuated joints)
        actuator_vel = qvel[6:]  # Skip freejoint velocities
        if len(actuator_vel) != len(torques):
            actuator_vel = qvel[-len(torques):]
        return -2.5e-5 * np.sum(np.abs(torques) * np.abs(actuator_vel))

    def regu_joint_velocity(self, qvel):
        """Joint velocity penalty: -1e-4 * ||q̇||²"""
        return -1e-4 * np.sum(qvel ** 2)

    def regu_joint_pos_limits(self):
        """Joint position limits penalty."""
        penalty = 0.0
        for i in range(self.model.njnt):
            if self.model.jnt_limited[i]:
                qpos_adr = self.model.jnt_qposadr[i]
                q = self.data.qpos[qpos_adr]
                lower = self.model.jnt_range[i, 0]
                upper = self.model.jnt_range[i, 1]
                # Penalty for exceeding limits
                if q < lower:
                    penalty -= 1e2 * (lower - q) ** 2
                elif q > upper:
                    penalty -= 1e2 * (q - upper) ** 2
        return penalty

    def regu_joint_vel_limits(self):
        """Joint velocity limits penalty: -1 * sum of violations."""
        penalty = 0.0
        # Use a reasonable velocity limit
        vel_limit = 10.0  # rad/s
        for i in range(6, self.model.nv):  # Skip freejoint
            vel = abs(self.data.qvel[i])
            if vel > vel_limit:
                penalty -= (vel - vel_limit)
        return penalty

    def compute_regularization_reward(self, action, torques):
        """Compute total regularization reward."""
        reward = 0.0
        reward += self.regu_joint_acceleration(self.data.qacc)
        reward += self.regu_action_rate(action)
        reward += self.regu_smoothness(action)
        reward += self.regu_torques(torques)
        reward += self.regu_joint_power(torques, self.data.qvel)
        reward += self.regu_joint_velocity(self.data.qvel)
        reward += self.regu_joint_pos_limits()
        reward += self.regu_joint_vel_limits()
        return reward

    # =========================================================================
    # (d) POST-TASK REWARDS (w^post = 1)
    # =========================================================================

    def post_base_angular_velocity(self):
        """Post-task: low angular velocity after standing."""
        h_base = self._get_base_height()
        if h_base <= self.H_STAGE2:
            return 0.0
        omega_xy = self._get_base_angular_velocity()[:2]
        reward = np.exp(-2 * np.linalg.norm(omega_xy) ** 2)
        return 10.0 * reward

    def post_base_linear_velocity(self):
        """Post-task: low linear velocity after standing."""
        h_base = self._get_base_height()
        if h_base <= self.H_STAGE2:
            return 0.0
        v_xy = self._get_base_linear_velocity()[:2]
        reward = np.exp(-5 * np.linalg.norm(v_xy) ** 2)
        return 10.0 * reward

    def post_base_orientation(self):
        """Post-task: upright orientation after standing."""
        h_base = self._get_base_height()
        if h_base <= self.H_STAGE2:
            return 0.0
        orientation = self._get_base_orientation()
        reward = np.exp(-5 * (1 - orientation) ** 2)
        return 10.0 * reward

    def post_base_height(self):
        """Post-task: maintain target height after standing."""
        h_base = self._get_base_height()
        if h_base <= self.H_STAGE2:
            return 0.0
        target_height = 1.28  # Standing height from XML
        reward = np.exp(-20 * (h_base - target_height) ** 2)
        return 10.0 * reward

    def post_feet_parallel(self):
        """Post-task: feet parallel to each other."""
        h_base = self._get_base_height()
        if h_base <= self.H_STAGE2:
            return 0.0
        h_right, h_left = self._get_foot_heights()
        height_diff = abs(h_right - h_left)
        reward = np.exp(-20 * np.clip(height_diff, 0.02, np.inf))
        return 2.5 * reward

    def compute_post_task_reward(self):
        """Compute total post-task reward."""
        reward = 0.0
        reward += self.post_base_angular_velocity()
        reward += self.post_base_linear_velocity()
        reward += self.post_base_orientation()
        reward += self.post_base_height()
        reward += self.post_feet_parallel()
        return reward

    # =========================================================================
    # MAIN REWARD COMPUTATION
    # =========================================================================

    def compute_reward(self, action, torques):
        """
        Compute total reward with all components.

        Returns:
            tuple: (total_reward, info_dict)
        """
        # Compute each reward group
        r_task = self.compute_task_reward()
        r_style = self.compute_style_reward()
        r_regu = self.compute_regularization_reward(action, torques)
        r_post = self.compute_post_task_reward()

        # Apply group weights
        total_reward = (
            self.W_TASK * r_task +
            self.W_STYLE * r_style +
            self.W_REGU * r_regu +
            self.W_POST * r_post
        )

        # Update action history
        self._prev_prev_action = self._prev_action
        self._prev_action = action.copy() if action is not None else None

        # Build info dict
        info = {
            "reward_task": r_task,
            "reward_style": r_style,
            "reward_regu": r_regu,
            "reward_post": r_post,
            "reward_task_weighted": self.W_TASK * r_task,
            "reward_style_weighted": self.W_STYLE * r_style,
            "reward_regu_weighted": self.W_REGU * r_regu,
            "reward_post_weighted": self.W_POST * r_post,
            "head_height": self._get_head_height(),
            "base_height": self._get_base_height(),
            "pelvis_height": self.data.xpos[self._pelvis_id, 2],
            "base_orientation": self._get_base_orientation(),
            "feet_contact_count": self._get_feet_contact_count(),
        }

        return total_reward, info

    def reset(self):
        """Reset reward calculator state."""
        self._prev_action = None
        self._prev_prev_action = None
