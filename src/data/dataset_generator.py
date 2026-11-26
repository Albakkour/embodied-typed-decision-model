import json
import shutil
from collections import Counter
from pathlib import Path

import torch

from libero.libero import benchmark, get_libero_path

from src.actions.candidate_generator import generate_candidate_actions
from src.data.schema import DecisionSample
from src.evaluation.action_quality import score_candidate_outcomes
from src.simulation.physics_preview import run_candidate_preview


def read_existing_samples(dataset_file):
    """Read existing IDs so generation can resume safely."""

    existing_ids = set()

    if not dataset_file.exists():
        return existing_ids

    with open(dataset_file, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            existing_ids.add(record["sample_id"])

    return existing_ids


def print_dataset_summary(dataset_file):
    """Summarize generated samples and teacher-label distribution."""

    label_counts = Counter()
    eligible = 0
    ambiguous = 0
    total = 0

    with open(dataset_file, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            sample = json.loads(line)
            total += 1

            label = sample["label"]

            if label["training_eligible"]:
                eligible += 1
                label_counts[label["target_choice"]] += 1
            else:
                ambiguous += 1

    print("\n========== DATASET SUMMARY ==========")

    print("Total samples:", total)
    print("Eligible samples:", eligible)
    print("Ambiguous samples:", ambiguous)

    print("\nEligible label distribution:")

    for action_id, count in label_counts.most_common():
        print(f"  {action_id}: {count}")

    if eligible:
        print("\nTop-action proportion:")

        for action_id, count in label_counts.most_common():
            print(f"  {action_id}: {count / eligible:.2%}")


def generate_dataset(
    task_ids=(0,),
    num_initial_states=10,
    translation_magnitude=0.25,
    horizon_steps=8,
    output_dir="data/processed/libero_s1_pilot",
    seed=0,
):
    """
    Generate LIBERO S1 candidate-selection samples.

    Samples are saved incrementally to JSONL.
    Existing sample IDs are skipped on subsequent runs.
    """

    output_dir = Path(output_dir)

    image_dir = output_dir / "images"
    preview_dir = output_dir / "previews"

    image_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = output_dir / "dataset.jsonl"

    existing_ids = read_existing_samples(dataset_file)

    suite = benchmark.get_benchmark_dict()["libero_10"]()

    candidates = generate_candidate_actions(
        translation_magnitude=translation_magnitude,
        horizon_steps=horizon_steps,
    )

    generated = 0
    skipped = 0

    # Open in append mode, preserving earlier samples.
    with open(dataset_file, "a", encoding="utf-8") as dataset:
        for task_id in task_ids:
            task = suite.get_task(task_id)

            bddl_file = (
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            )

            init_file = (
                Path(get_libero_path("init_states"))
                / task.problem_folder
                / task.init_states_file
            )

            if not bddl_file.exists():
                raise FileNotFoundError(bddl_file)

            if not init_file.exists():
                raise FileNotFoundError(init_file)

            initial_states = torch.load(
                init_file,
                map_location="cpu",
                weights_only=False,
            )

            count = min(
                num_initial_states,
                len(initial_states),
            )

            print("\n" + "=" * 65)
            print("TASK:", task.name)
            print("Instruction:", task.language)
            print("Initial states requested:", count)
            print("=" * 65)

            for state_id in range(count):
                sample_id = f"libero_10_task_{task_id:02d}_state_{state_id:03d}"

                if sample_id in existing_ids:
                    print("Skipping existing sample:", sample_id)
                    skipped += 1
                    continue

                print("\nGenerating:", sample_id)

                initial_state = initial_states[state_id].copy()

                outcomes = []

                # Every candidate gets its own environment.
                for candidate in candidates:
                    candidate_dir = preview_dir / sample_id / candidate.action_id

                    outcome = run_candidate_preview(
                        bddl_file=bddl_file,
                        initial_state=initial_state,
                        candidate=candidate,
                        output_dir=candidate_dir,
                        seed=seed,
                    )

                    outcomes.append(outcome)

                    print(
                        "  Completed:",
                        candidate.action_id,
                    )

                # Compute heuristic teacher labels.
                evaluation = score_candidate_outcomes(
                    outcomes,
                    completion_bonus=100.0,
                    minimum_score_margin=1.0,
                )

                observation_image = image_dir / f"{sample_id}.png"

                shutil.copy2(
                    outcomes[0].image_before,
                    observation_image,
                )

                candidate_records = [
                    {
                        "action_id": candidate.action_id,
                        "name": candidate.name,
                        "description": candidate.description,
                        "vector": candidate.to_numpy().tolist(),
                        "horizon_steps": candidate.horizon_steps,
                    }
                    for candidate in candidates
                ]

                label = {
                    "target_choice": evaluation["target_choice"],
                    "confident": evaluation["confident"],
                    "training_eligible": evaluation["confident"],
                    "score_margin": evaluation["score_margin"],
                    "scores": evaluation["scores"],
                    "ranked_candidates": evaluation["ranked_candidates"],
                    "method": evaluation["label_method"],
                    "is_heuristic": True,
                    "is_ground_truth": False,
                }

                sample = DecisionSample(
                    schema_version="1.0",
                    sample_id=sample_id,
                    task={
                        "suite": "libero_10",
                        "task_id": task_id,
                        "task_name": task.name,
                        "instruction": task.language,
                        "initial_state_id": state_id,
                        "goal_predicates": (outcomes[0].goal_predicates_before),
                    },
                    input={
                        "image_path": str(observation_image),
                        "image_width": 128,
                        "image_height": 128,
                        "robot_state": outcomes[0].robot_state_before,
                        "candidate_actions": candidate_records,
                    },
                    label=label,
                    teacher_outcomes=[
                        {
                            **outcome.to_dict(),
                            "quality_score": scored["quality_score"],
                            "score_components": scored["score_components"],
                        }
                        for outcome, scored in zip(
                            outcomes,
                            evaluation["scored_outcomes"],
                        )
                    ],
                    metadata={
                        "data_source": "LIBERO_simulated_candidate_selection",
                        "preview_duration_seconds": horizon_steps / 20.0,
                        "control_frequency_hz": 20,
                        "translation_magnitude": translation_magnitude,
                        "action_horizon_steps": horizon_steps,
                        "label_method": evaluation["label_method"],
                        "label_is_heuristic": True,
                        "seed": seed,
                    },
                )

                # Write and flush each sample immediately.
                dataset.write(
                    json.dumps(
                        sample.to_dict(),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                dataset.flush()

                existing_ids.add(sample_id)
                generated += 1

                print("  Target:", label["target_choice"])
                print("  Confident:", label["confident"])

    print("\nNew samples generated:", generated)
    print("Existing samples skipped:", skipped)
    print("Dataset:", dataset_file.resolve())

    print_dataset_summary(dataset_file)

    return dataset_file
