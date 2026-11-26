from typing import Any

import numpy as np

from src.data.schema import CandidatePreview


def _distance(a, b) -> float:
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        )
    )


def _approach_progress(
    outcome: CandidatePreview,
    object_name: str,
) -> float:
    """
    Return end-effector approach progress in millimetres.

    Positive: robot gets closer to the object.
    Negative: robot gets farther from the object.
    """

    object_key = f"{object_name}_pos"

    object_before = outcome.object_positions_before.get(object_key)
    object_after = outcome.object_positions_after.get(object_key)

    if object_before is None or object_after is None:
        return 0.0

    distance_before = _distance(
        outcome.eef_position_before,
        object_before,
    )

    distance_after = _distance(
        outcome.eef_position_after,
        object_after,
    )

    return (distance_before - distance_after) * 1000.0


def score_candidate_outcomes(
    outcomes: list[CandidatePreview],
    completion_bonus: float = 100.0,
    minimum_score_margin: float = 1.0,
) -> dict[str, Any]:
    """
    Score candidates using:

      1. Goal predicates newly satisfied.
      2. Approach improvement relative to hold_position.

    This is a provisional heuristic teacher, not a
    ground-truth optimal-action oracle.
    """

    if not outcomes:
        raise ValueError("No candidate outcomes supplied.")

    by_id = {outcome.candidate_id: outcome for outcome in outcomes}

    if len(by_id) != len(outcomes):
        raise ValueError("Duplicate candidate IDs.")

    if "hold_position" not in by_id:
        raise ValueError("A hold_position candidate is required as baseline.")

    baseline = by_id["hold_position"]

    # Identify objects whose goal predicates are not yet satisfied.
    active_objects = []

    for predicate in baseline.goal_predicates_before:
        if not predicate["satisfied"]:
            object_name = predicate["subject"]

            if object_name not in active_objects:
                active_objects.append(object_name)

    # Measure the baseline progress for each active target.
    baseline_progress = {
        object_name: _approach_progress(
            baseline,
            object_name,
        )
        for object_name in active_objects
    }

    scored_outcomes = []

    for outcome in outcomes:
        progress_by_object = {}

        for object_name in active_objects:
            raw_progress = _approach_progress(
                outcome,
                object_name,
            )

            progress_by_object[object_name] = (
                raw_progress - baseline_progress[object_name]
            )

        approach_score = (
            float(np.mean(list(progress_by_object.values())))
            if progress_by_object
            else 0.0
        )

        # Count newly satisfied individual goal predicates.
        before = outcome.goal_predicates_before
        after = outcome.goal_predicates_after

        newly_satisfied = 0

        for before_goal, after_goal in zip(before, after):
            if not before_goal["satisfied"] and after_goal["satisfied"]:
                newly_satisfied += 1

        goal_score = completion_bonus * newly_satisfied

        quality_score = goal_score + approach_score

        scored_outcomes.append(
            {
                **outcome.to_dict(),
                "score_components": {
                    "newly_satisfied_goals": newly_satisfied,
                    "goal_completion_score": goal_score,
                    "approach_progress_relative_to_hold_mm": (progress_by_object),
                    "mean_relative_approach_progress_mm": (approach_score),
                },
                "quality_score": float(quality_score),
            }
        )

    ranked = sorted(
        scored_outcomes,
        key=lambda x: (
            -x["quality_score"],
            x["candidate_id"],
        ),
    )

    best = ranked[0]

    second_score = ranked[1]["quality_score"] if len(ranked) > 1 else float("-inf")

    score_margin = best["quality_score"] - second_score

    confident = score_margin >= minimum_score_margin

    # Do not assign a training label when the ranking is ambiguous.
    target_choice = best["candidate_id"] if confident else None

    return {
        "target_choice": target_choice,
        "confident": confident,
        "score_margin": float(score_margin),
        "scores": {
            outcome["candidate_id"]: outcome["quality_score"]
            for outcome in scored_outcomes
        },
        "ranked_candidates": [outcome["candidate_id"] for outcome in ranked],
        "scored_outcomes": scored_outcomes,
        "label_method": ("baseline_corrected_approach_plus_goal_completion_v1"),
        "label_is_heuristic": True,
    }
