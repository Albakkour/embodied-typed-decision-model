from src.actions.candidate_generator import (
    generate_candidate_actions,
)

from src.simulation.physics_preview import (
    run_candidate_previews,
)


def main():

    # Experimental configuration
    candidates = generate_candidate_actions(
        translation_magnitude=0.25,
        horizon_steps=8,
    )

    results = run_candidate_previews(
        task_id=0,
        initial_state_id=0,
        candidates=candidates,
        output_dir="data/debug/physics_previews_scale_025",
        seed=0,
    )

    print("\n========== FINAL SUMMARY ==========")

    for result in results:
        print(f"\n{result.candidate_id}")

        print(
            "EEF displacement:",
            result.eef_displacement,
        )

        print(
            "Initial gripper:",
            result.initial_gripper_qpos,
        )

        print(
            "Final gripper:",
            result.final_gripper_qpos,
        )

        print("Reward:", result.reward_sum)
        print("Success:", result.done)


if __name__ == "__main__":
    main()
