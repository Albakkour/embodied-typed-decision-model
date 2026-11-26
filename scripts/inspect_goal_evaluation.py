from pathlib import Path
import inspect
import pprint

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

    states = torch.load(
        init_file,
        map_location="cpu",
        weights_only=False,
    )

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=128,
        camera_widths=128,
    )

    try:
        env.reset()
        obs = env.set_init_state(states[0])

        task_env = env.env
        problem = task_env.parsed_problem

        print("\n========== GOAL PREDICATES ==========")

        pprint.pp(problem["goal_state"])

        print("\n========== PARSED PROBLEM KEYS ==========")

        print(list(problem.keys()))

        print("\n========== REGION DEFINITIONS ==========")

        for key, value in problem.items():
            if "region" in key.lower():
                print(f"\n{key}:")
                pprint.pp(value)

        print("\n========== CONTAINMENT EVALUATION ==========")

        print(inspect.getsource(type(task_env)._eval_predicate))

        print("\n========== SUCCESS CHECK ==========")

        print("Initial success:", env.check_success())

    finally:
        env.close()


if __name__ == "__main__":
    main()
