import json
from pathlib import Path

import numpy as np


RESULTS_PATH = Path("data/debug/physics_previews_scale_025/task_0_state_0/results.json")


TARGET_OBJECTS = [
    "alphabet_soup_1_pos",
    "tomato_sauce_1_pos",
]


def calculate_distance(a, b):
    return float(
        np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    )


def main():

    with open(RESULTS_PATH, encoding="utf-8") as file:
        results = json.load(file)

    baseline = next(
        result for result in results if result["candidate_id"] == "hold_position"
    )

    baseline_progress = {}

    for object_name in TARGET_OBJECTS:
        d_before = calculate_distance(
            baseline["initial_eef_position"],
            baseline["object_positions_before"][object_name],
        )

        d_after = calculate_distance(
            baseline["final_eef_position"],
            baseline["object_positions_after"][object_name],
        )

        baseline_progress[object_name] = d_before - d_after

    print("=" * 75)
    print("TARGET APPROACH ANALYSIS")
    print("=" * 75)

    print("\nPositive progress means getting closer to the object.")

    for result in results:
        print(f"\nCandidate: {result['candidate_id']}")

        for object_name in TARGET_OBJECTS:
            d_before = calculate_distance(
                result["initial_eef_position"],
                result["object_positions_before"][object_name],
            )

            d_after = calculate_distance(
                result["final_eef_position"],
                result["object_positions_after"][object_name],
            )

            progress = d_before - d_after

            relative_progress = progress - baseline_progress[object_name]

            print(f"\n{object_name}")

            print(f"  Initial distance: {d_before:.5f} m")
            print(f"  Final distance:   {d_after:.5f} m")

            print(f"  Approach progress: {progress * 1000:.3f} mm")

            print(f"  Progress relative to hold: {relative_progress * 1000:.3f} mm")

        # Gripper movement
        initial_gripper = np.asarray(result["initial_gripper_qpos"])

        final_gripper = np.asarray(result["final_gripper_qpos"])

        print(
            "\nGripper movement:",
            final_gripper - initial_gripper,
        )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
