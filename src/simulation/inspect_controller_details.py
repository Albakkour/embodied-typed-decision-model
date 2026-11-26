from pathlib import Path
import inspect

import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.controllers import osc


def main():
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    init_file = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )

    states = torch.load(
        init_file,
        map_location="cpu",
        weights_only=False,
    )

    env = None

    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=128,
            camera_widths=128,
        )

        env.seed(0)
        env.reset()
        env.set_init_state(states[0])

        robot = env.robots[0]
        controller = robot.controller
        gripper = robot.gripper
        base_env = env.env

        print("\n========== OSC CONFIGURATION ==========")

        attributes = [
            "use_delta",
            "impedance_mode",
            "use_ori",
            "position_limits",
            "orientation_limits",
            "control_dim",
            "control_freq",
        ]

        for name in attributes:
            print(f"{name}: {getattr(controller, name, 'Unavailable')}")

        print("\n========== ACTION SCALING ==========")

        print(inspect.getsource(controller.scale_action))

        print("\n========== POSITION GOAL FUNCTION ==========")

        print(inspect.getsource(osc.set_goal_position))

        print("\n========== GRIPPER ==========")

        print("Speed:", gripper.speed)
        print("Current action:", gripper.current_action)
        print("DOF:", gripper.dof)

        print("\n========== SIMULATION TIMING ==========")

        print("Control frequency:", base_env.control_freq)
        print("Simulation timestep:", base_env.sim.model.opt.timestep)
        print("Controller timestep:", 1.0 / base_env.control_freq)

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
