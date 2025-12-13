### 🛠️ The Projects (Menu for Participants)

Divide your 4 experts into **2 Teams** or **4 Solo Tracks**. I recommend **4 Solo Tracks** that merge at the end of the day, as high-skill individual contributors often move faster alone on distinct modules.

#### Track 1: The Morphologist (Procedural Robot Generator)

**Objective:** Build the tooling to load the Humanoid XML, procedurally scale it (height, mass, actuators) based on a config, and save the new XML.

  * **Why:** Solves the "10m Robot" PRD requirement.
  * **Input:** `humanoid.xml`, scaling factor (e.g., 2.0).
  * **Output:** `humanoid_scaled.xml` where mesh scales, joint positions, and actuator gains are multiplied correctly.
  * **Success Criteria:** A script that launches a viewer showing a normal Humanoid next to a giant Humanoid.
  * **Key Tool:** `dm_control.mjcf` (best for parsing/editing MJCF programmatically).

#### Track 2: The Pilot (Teleop & Kinematics)

**Objective:** Build the "Input Interface" for the simulator. Map a PS5/Xbox controller to the robot's joints.

  * **Why:** We need to debug physics and test "fighting moves" manually before training RL.
  * **Input:** Joystick signals.
  * **Output:** PD targets sent to `mujoco.MjData.ctrl`.
  * **Challenge:** Implement a basic "Inverse Kinematics" (IK) feature where the joystick moves the *hand* (end-effector) rather than just rotating joints.
  * **Success Criteria:** A human can pick up the controller and make the Humanoid throw a punch in the sim.

#### Track 3: The Trainer (RL Locomotion Loop)

**Objective:** Set up the standard `Gymnasium` adapter for MuJoCo and train a PPO policy to stand up and resist gravity.

  * **Why:** Core Phase 0 milestone.
  * **Task:**
    1.  Subclass `gymnasium.Env`.
    2.  Define `observation_space` (qpos, qvel, IMU).
    3.  Define `action_space` (motor positions).
    4.  Connect `Stable-Baselines3` PPO.
  * **Success Criteria:** A training curve that goes up, and a saved video of the Humanoid standing for \>10 seconds without falling.

#### Track 4: The Architect (Multi-Agent Arena)

**Objective:** Solve the "Namespace Collision" problem to spawn TWO robots in one scene.

  * **Why:** MuJoCo XMLs often have hardcoded names (`torso`, `leg`). To have two robots, you need to prefix them (`p1/torso`, `p2/torso`).
  * **Task:** Write a script that takes a robot XML, prefixes all body/joint/actuator names, and merges two of them into a single `arena.xml`.
  * **Success Criteria:** A sim window showing two Humanoids facing each other.

-----

### 🚀 Zero-Start Instructions (Copy-Paste for Participants)

#### Step 1: Clone & Install

```bash
git clone https://github.com/S3-Studios/vrfa-hackathon-2025.git
cd vrfa-hackathon-2025
conda create -n vrfa-hackathon-2025 python=3.10 -y
conda activate vrfa-hackathon-2025
pip install -r requirements.txt
```

#### Step 2: Pick Your Starter Code

**For Track 1: The Morphologist (Procedural Robot Generator)**
*Starter File:* [`tracks/track1_morphologist/scale_robot.py`](tracks/track1_morphologist/scale_robot.py)

Run the script to generate a giant Humanoid:
```bash
python tracks/track1_morphologist/scale_robot.py
```

**For Track 2: The Pilot (Teleop & Kinematics)**
*Starter File:* [`tracks/track2_pilot/teleop.py`](tracks/track2_pilot/teleop.py)

Run the teleoperation script (requires a window):
```bash
python tracks/track2_pilot/teleop.py
```

**For Track 3: The Trainer (RL Locomotion Loop)**
*Starter File:* [`tracks/track3_trainer/train_env.py`](tracks/track3_trainer/train_env.py)

Run the training loop test:
```bash
python tracks/track3_trainer/train_env.py
```

**For Track 4: The Architect (Multi-Agent Arena)**
*Starter File:* [`tracks/track4_architect/merge_robots.py`](tracks/track4_architect/merge_robots.py)

Run the merge script to create a multi-agent arena:
```bash
python tracks/track4_architect/merge_robots.py
```

-----

### 💡 Advice for the Day

1.  **Ignore Rendering:** Tell Track 3 (RL) to run headless (no viewer) during training. Rendering slows down MuJoCo training by 100x. Only render during "eval".
2.  **Focus on "The Merge":** At 3:00 PM, ask Track 1 (Scaling) to give their XML generator to Track 3 (RL). Can the RL agent train on a 2m robot just by changing the config? That is the **Phase 0 Golden Moment**.
3.  **Sim-to-Sim validation:** If Track 2 (Teleop) gets a working punch, record the joint angles. Pass that data to Track 3 as a "reference motion" to see if the RL can mimic it (basic Imitation Learning).

**Deliverable by 5 PM:** A repo where you can run `python train_agent.py --scale 2.0` and watch a giant robot learn to stand.