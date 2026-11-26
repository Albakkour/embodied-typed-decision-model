from pathlib import Path

import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.actions.candidate_generator import (
    generate_candidate_actions,
)
from src.simulation.physics_preview import (
    run_candidate_preview,
)


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

    initial_state = states[0].copy()

    print("\n========== INITIAL GEOMETRY ==========")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=128,
        camera_widths=128,
    )

    try:
        env.reset()
        obs = env.set_init_state(initial_state)

        eef = obs["robot0_eef_pos"]

        for object_name in [
            "alphabet_soup_1_pos",
            "tomato_sauce_1_pos",
        ]:
            object_pos = obs[object_name]

            import numpy as np

            distance = np.linalg.norm(eef - object_pos)

            print(f"EEF to {object_name}: {distance:.4f} m")

    finally:
        env.close()

    print("\n========== ACTION SCALE EXPERIMENT ==========")

    magnitudes = [0.1, 0.25, 0.5, 1.0]

    output_root = Path("data/debug/action_scales")
    output_root.mkdir(parents=True, exist_ok=True)

    for magnitude in magnitudes:
        candidates = generate_candidate_actions(
            translation_magnitude=magnitude,
            horizon_steps=8,
        )

        candidate = next(c for c in candidates if c.action_id == "move_x_positive")

        print(f"\nTesting magnitude: {magnitude}")

        result = run_candidate_preview(
            task=task,
            bddl_file=bddl_file,
            initial_state=initial_state,
            candidate=candidate,
            output_dir=output_root / f"scale_{magnitude}",
            seed=0,
        )

        print("Action:", result.action)
        print("EEF displacement:", result.eef_displacement)

        print(
            "Displacement magnitude:", f"{result.eef_displacement_norm_m * 1000:.3f} mm"
        )

    print("\nAll action-scale experiments completed.")


if __name__ == "__main__":
    main()
