from pathlib import Path

import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    print("=" * 60)
    print("INITIALIZING LIBERO SIMULATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load the LIBERO-10 benchmark
    # --------------------------------------------------

    benchmark_dict = benchmark.get_benchmark_dict()

    suite = benchmark_dict["libero_10"]()

    task_id = 0
    task = suite.get_task(task_id)

    print("\nTASK INFORMATION")
    print("Suite:", "libero_10")
    print("Task ID:", task_id)
    print("Task name:", task.name)
    print("Instruction:", task.language)

    # --------------------------------------------------
    # 2. Locate the BDDL task definition
    # --------------------------------------------------

    bddl_file = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )

    if not bddl_file.exists():
        raise FileNotFoundError(
            f"BDDL file not found: {bddl_file}"
        )

    print("\nBDDL FILE")
    print(bddl_file)

    # --------------------------------------------------
    # 3. Load predefined initial states
    # --------------------------------------------------

    init_states_file = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )

    if not init_states_file.exists():
        raise FileNotFoundError(
            f"Initial states not found: {init_states_file}"
        )

    # LIBERO initial states are trusted simulator files.
    # PyTorch 2.6+ requires weights_only=False for
    # these NumPy-containing serialized objects.
    init_states = torch.load(
        init_states_file,
        map_location="cpu",
        weights_only=False,
    )

    print("\nINITIAL STATES")
    print("Number of initial states:", len(init_states))
    print("Initial state type:", type(init_states[0]))
    print("Initial state shape:", init_states[0].shape)

    # --------------------------------------------------
    # 4. Create the MuJoCo simulation
    # --------------------------------------------------

    env = None

    try:
        print("\nCREATING ROBOT ENVIRONMENT")

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=128,
            camera_widths=128,
        )

        env.seed(0)

        # Initialize the simulation.
        obs = env.reset()

        print("Environment initialized successfully.")

        # --------------------------------------------------
        # 5. Load a reproducible initial state
        # --------------------------------------------------

        obs = env.set_init_state(init_states[0])

        print("Initial state applied successfully.")

        # --------------------------------------------------
        # 6. Inspect observations
        # --------------------------------------------------

        print("\nOBSERVATION INFORMATION")

        print("Available observation keys:")

        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(
                    f"  {key}: shape={value.shape}, "
                    f"dtype={value.dtype}"
                )
            else:
                print(f"  {key}: {type(value)}")

        # Main external camera observation.
        image = obs["agentview_image"]

        print("\nCAMERA INFORMATION")
        print("Image shape:", image.shape)
        print("Image dtype:", image.dtype)
        print("Minimum pixel:", image.min())
        print("Maximum pixel:", image.max())

        # --------------------------------------------------
        # 7. Inspect robot state
        # --------------------------------------------------

        print("\nROBOT STATE")

        robot_state_keys = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
        ]

        for key in robot_state_keys:
            if key in obs:
                print(f"{key}: {obs[key]}")

        # --------------------------------------------------
        # 8. Execute one test action
        # --------------------------------------------------

        print("\nEXECUTING TEST ACTION")

        action = np.zeros(7, dtype=np.float32)

        print("Action:", action)

        obs, reward, done, info = env.step(action)

        print("Action executed successfully.")
        print("Reward:", reward)
        print("Done:", done)

        print("\nUPDATED OBSERVATION")
        print("Image shape:", obs["agentview_image"].shape)

        print("\n" + "=" * 60)
        print("FIRST LIBERO SIMULATION COMPLETED SUCCESSFULLY")
        print("=" * 60)

    finally:
        if env is not None:
            env.close()
            print("\nSimulation environment closed.")


if __name__ == "__main__":
    main()
