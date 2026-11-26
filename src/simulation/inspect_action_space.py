from pathlib import Path

import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)

    bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )

    init_file = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )

    init_states = torch.load(
        init_file,
        map_location="cpu",
        weights_only=False,
    )

    env = None

    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=128,
            camera_widths=128,
        )

        env.seed(0)
        env.reset()

        obs = env.set_init_state(init_states[0])

        # Access the underlying robosuite environment.
        base_env = env.env

        print("\n========== ACTION SPACE ==========")

        action_low, action_high = base_env.action_spec

        print("Action dimension:", base_env.action_dim)
        print("Action lower bounds:", action_low)
        print("Action upper bounds:", action_high)

        print("\n========== ROBOT CONTROLLER ==========")

        robot = env.robots[0]
        controller = robot.controller

        print("Robot:", type(robot).__name__)
        print("Controller:", type(controller).__name__)

        for attribute in [
            "name",
            "control_dim",
            "input_min",
            "input_max",
            "output_min",
            "output_max",
            "control_freq",
        ]:
            print(f"{attribute}: {getattr(controller, attribute, 'Not available')}")

        print("\n========== ROBOT AND GRIPPER ==========")

        print("Robot action limits:", robot.action_limits)

        gripper = robot.gripper

        print("Gripper:", type(gripper).__name__)
        print("Gripper DOF:", gripper.dof)

        print("\n========== ENVIRONMENT ==========")

        print("Control frequency:", base_env.control_freq)
        print("Action dimension:", base_env.action_dim)

        print("\n========== OBSERVATION ==========")

        print("Agent view:", obs["agentview_image"].shape)
        print(
            "Wrist camera:",
            obs["robot0_eye_in_hand_image"].shape,
        )

        print("\nInspection completed successfully.")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
