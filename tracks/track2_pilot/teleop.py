import mujoco
import mujoco.viewer
import numpy as np
import os
import time

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xml_path = os.path.join(base_dir, "assets", "humanoid", "humanoid.xml")
    
    print(f"Loading model from {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    print("Launching passive viewer... (Close window to exit)")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        while viewer.is_running():
            # TODO: Hook up pygame joystick here
            # control_signal = joystick.get_axis()
            
            # Simple test: Sine wave on an actuator (index 4 might be an arm joint)
            # Check nu (number of actuators)
            if m.nu > 4:
                d.ctrl[4] = np.sin(time.time() - start_time) 
            
            mujoco.mj_step(m, d)
            viewer.sync()
            
            # Slow down to real-time (approx)
            time_until_next_step = m.opt.timestep - (time.time() - start_time) % m.opt.timestep
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
