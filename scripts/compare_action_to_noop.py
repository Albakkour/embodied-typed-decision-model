from pathlib import Path

import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.actions.candidate_generator import (
    generate_candidate_actions,
)


TASK_ID = 0
HORIZON = 8


def run_trial(task, init_state, action, label, bddl_file):
    """Run one independent trial from the same initial state."""

    env = None

    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=128,
            camera_widths=128,
        )

        env.seed(0)
        env.reset()

        obs = env.set_init_state(init_state)

        robot = env.robots[0]
        controller = robot.controller

        # Compare LIBERO's observation with the controller cache.
        observed_position = obs["robot0_eef_pos"].copy()
        cached_before = np.array(controller.ee_pos).copy()

        # Explicitly refresh controller state after restoring
        # the simulator's initial state.
        controller.update(force=True)

        cached_after = np.array(controller.ee_pos).copy()
        # Synchronize the controller's desired pose with the
        # robot's actual pose after restoring the initial state.
        controller.set_goal(
            np.zeros(controller.control_dim),
            set_pos=controller.ee_pos.copy(),
            set_ori=controller.ee_ori_mat.copy(),
        )

        print("Controller goal position:", controller.goal_pos)
        print("Actual position:", controller.ee_pos)

        print(
            "Orientation target error:",
            np.linalg.norm(controller.goal_ori - controller.ee_ori_mat),
        )

        print(f"\n========== {label} ==========")
        print("Observation EEF:", observed_position)
        print("Controller cache before refresh:", cached_before)
        print("Controller cache after refresh:", cached_after)

        eef_positions = [observed_position.copy()]
        soup_positions = [obs["alphabet_soup_1_pos"].copy()]
        tomato_positions = [obs["tomato_sauce_1_pos"].copy()]

        total_reward = 0.0

        for step in range(HORIZON):
            obs, reward, done, info = env.step(action)

            total_reward += float(reward)

            eef_positions.append(obs["robot0_eef_pos"].copy())

            soup_positions.append(obs["alphabet_soup_1_pos"].copy())

            tomato_positions.append(obs["tomato_sauce_1_pos"].copy())

            print(f"Step {step + 1}: EEF={obs['robot0_eef_pos']}")

            if done:
                break

        eef_positions = np.asarray(eef_positions)
        soup_positions = np.asarray(soup_positions)
        tomato_positions = np.asarray(tomato_positions)

        result = {
            "eef_initial": eef_positions[0],
            "eef_final": eef_positions[-1],
            "eef_displacement": (eef_positions[-1] - eef_positions[0]),
            "eef_positions": eef_positions,
            "soup_displacement": (soup_positions[-1] - soup_positions[0]),
            "tomato_displacement": (tomato_positions[-1] - tomato_positions[0]),
            "reward": total_reward,
        }

        print("\nTrial displacement:")
        print(result["eef_displacement"])

        print("Soup displacement:")
        print(result["soup_displacement"])

        print("Tomato displacement:")
        print(result["tomato_displacement"])

        print("Total reward:", total_reward)

        return result

    finally:
        if env is not None:
            env.close()


def main():
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(TASK_ID)

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

    initial_state = init_states[0].copy()

    candidates = generate_candidate_actions()

    positive_x = next(c for c in candidates if c.action_id == "move_x_positive")

    noop = next(c for c in candidates if c.action_id == "hold_position")

    baseline = run_trial(
        task,
        initial_state,
        noop.to_numpy(),
        "NO-OP BASELINE",
        bddl_file,
    )

    positive = run_trial(
        task,
        initial_state,
        positive_x.to_numpy(),
        "POSITIVE X",
        bddl_file,
    )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print("Baseline displacement:", baseline["eef_displacement"])
    print("Positive-X displacement:", positive["eef_displacement"])

    difference = positive["eef_final"] - baseline["eef_final"]

    print("\nPositive-X effect relative to baseline:")
    print(difference)

    print("\nInterpretation:")
    print("Positive X requested:", positive_x.vector)
    print("Baseline reward:", baseline["reward"])
    print("Positive-X reward:", positive["reward"])


if __name__ == "__main__":
    main()
