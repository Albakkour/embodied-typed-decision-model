from pathlib import Path
import inspect

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

    env = None

    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=128,
            camera_widths=128,
        )

        env.reset()
        env.set_init_state(states[0])

        print("\n========== TASK ==========")
        print("Name:", task.name)
        print("Instruction:", task.language)

        print("\n========== ENVIRONMENT TYPES ==========")

        current = env
        visited = set()

        for depth in range(6):
            if id(current) in visited:
                break

            visited.add(id(current))

            print(
                f"\nDepth {depth}:",
                type(current).__module__,
                type(current).__name__,
            )

            print("Relevant methods:")

            for name in dir(current):
                if any(
                    word in name.lower()
                    for word in [
                        "success",
                        "goal",
                        "reward",
                    ]
                ):
                    attribute = getattr(current, name, None)

                    if callable(attribute):
                        print(" ", name)

                        try:
                            print(inspect.getsource(getattr(type(current), name)))
                        except (OSError, TypeError, AttributeError):
                            print("  Source unavailable")

            next_env = getattr(current, "env", None)

            if next_env is None or next_env is current:
                break

            current = next_env

        print("\n========== INITIAL SUCCESS STATUS ==========")

        for name in ["_check_success", "check_success"]:
            method = getattr(env, name, None)

            if callable(method):
                print(name, ":", method())

        print("\nInspection completed.")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
