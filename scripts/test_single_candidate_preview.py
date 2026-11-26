from pathlib import Path

import numpy as np
import torch
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.actions.candidate_generator import (
    generate_candidate_actions,
)


def main():
    task_id = 0

    # 1. Load benchmark and task
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(task_id)

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

    # 2. Select one candidate
    candidates = generate_candidate_actions()

    candidate = next(
        action for action in candidates if action.action_id == "move_x_positive"
    )

    print("=" * 60)
    print("SINGLE CANDIDATE PHYSICS PREVIEW")
    print("=" * 60)

    print("Task:", task.name)
    print("Candidate:", candidate.name)
    print("Action:", candidate.to_numpy())
    print("Horizon:", candidate.horizon_steps)

    # 3. Initialize environment
    env = None
    output_dir = Path("data/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=128,
            camera_widths=128,
        )

        env.seed(0)
        env.reset()

        obs = env.set_init_state(init_states[0])

        # 4. Record initial measurements
        eef_before = obs["robot0_eef_pos"].copy()

        soup_before = obs["alphabet_soup_1_pos"].copy()
        tomato_before = obs["tomato_sauce_1_pos"].copy()
        basket_before = obs["basket_1_pos"].copy()

        def distance_to_basket(object_position):
            return float(np.linalg.norm(object_position - basket_before))

        soup_distance_before = distance_to_basket(soup_before)
        tomato_distance_before = distance_to_basket(tomato_before)

        Image.fromarray(obs["agentview_image"]).save(
            output_dir / "candidate_before.png"
        )

        print("\nINITIAL STATE")
        print("EEF position:", eef_before)
        print("Soup position:", soup_before)
        print("Tomato position:", tomato_before)
        print("Basket position:", basket_before)

        # 5. Execute the candidate for its full horizon
        total_reward = 0.0
        done = False

        print("\nEXECUTING CANDIDATE")

        for step in range(candidate.horizon_steps):
            obs, reward, done, info = env.step(candidate.to_numpy())

            total_reward += float(reward)

            print(
                f"Step {step + 1}/{candidate.horizon_steps}: "
                f"reward={reward}, done={done}"
            )

            if done:
                break

        # 6. Record final measurements
        eef_after = obs["robot0_eef_pos"].copy()

        soup_after = obs["alphabet_soup_1_pos"].copy()
        tomato_after = obs["tomato_sauce_1_pos"].copy()

        soup_distance_after = distance_to_basket(soup_after)
        tomato_distance_after = distance_to_basket(tomato_after)

        Image.fromarray(obs["agentview_image"]).save(output_dir / "candidate_after.png")

        # 7. Calculate physical changes
        eef_displacement = eef_after - eef_before

        print("\n========== RESULTS ==========")

        print("Initial EEF:", eef_before)
        print("Final EEF:", eef_after)
        print("EEF displacement:", eef_displacement)
        print("EEF displacement magnitude:", np.linalg.norm(eef_displacement))

        print("\nOBJECT MOVEMENT")

        print("Soup displacement:", soup_after - soup_before)

        print("Tomato displacement:", tomato_after - tomato_before)

        print("\nOBJECT-TO-BASKET DISTANCE")

        print(f"Soup: {soup_distance_before:.4f} -> {soup_distance_after:.4f} m")

        print(f"Tomato: {tomato_distance_before:.4f} -> {tomato_distance_after:.4f} m")

        print("\nTotal reward:", total_reward)
        print("Done:", done)
        print("Info keys:", list(info.keys()))

        print("\nImages saved in:", output_dir.resolve())

        print("\nPreview completed successfully.")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
