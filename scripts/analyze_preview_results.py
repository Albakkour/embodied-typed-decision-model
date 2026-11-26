import json
from pathlib import Path

import numpy as np


RESULTS = Path("data/debug/physics_previews/task_0_state_0/results.json")


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def main():
    with open(RESULTS, encoding="utf-8") as f:
        results = json.load(f)

    baseline = next(r for r in results if r["candidate_id"] == "hold_position")

    basket_before = baseline["object_positions_before"]["basket_1_pos"]

    basket_after = baseline["object_positions_after"]["basket_1_pos"]

    target_objects = [
        "alphabet_soup_1_pos",
        "tomato_sauce_1_pos",
    ]

    print("=" * 85)
    print("OBJECT MOVEMENT AND TASK PROGRESS ANALYSIS")
    print("=" * 85)

    for result in results:
        before = result["object_positions_before"]
        after = result["object_positions_after"]

        print(f"\nCandidate: {result['candidate_id']}")

        for object_name in target_objects:
            initial = np.asarray(before[object_name])
            final = np.asarray(after[object_name])

            movement_mm = np.linalg.norm(final - initial) * 1000

            distance_before = distance(initial, basket_before)

            distance_after = distance(final, basket_after)

            progress_mm = (distance_before - distance_after) * 1000

            print(
                f"{object_name}:"
                f"\n  Movement: {movement_mm:.4f} mm"
                f"\n  Basket distance before:"
                f" {distance_before:.4f} m"
                f"\n  Basket distance after:"
                f" {distance_after:.4f} m"
                f"\n  Distance improvement:"
                f" {progress_mm:.4f} mm"
            )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
