from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CandidateAction:
    """
    A discrete candidate action for the LIBERO robot.

    The action contains six OSC_POSE commands and one
    gripper command.
    """

    action_id: str
    name: str
    description: str
    vector: Tuple[float, ...]
    horizon_steps: int

    def to_numpy(self) -> np.ndarray:
        """Convert the action into a NumPy vector."""
        return np.asarray(self.vector, dtype=np.float64)

    @property
    def dimension(self) -> int:
        return len(self.vector)


def generate_candidate_actions(
    translation_magnitude: float = 0.1,
    horizon_steps: int = 8,
) -> list[CandidateAction]:
    """
    Generate an initial set of nine robot action primitives.

    Action format:
        [dx, dy, dz, d_rx, d_ry, d_rz, gripper]

    The six arm dimensions use normalized OSC commands.

    Gripper:
        -1 = open
         0 = hold
         1 = close

    Args:
        translation_magnitude:
            Normalized translation command magnitude.

        horizon_steps:
            Number of policy steps for each candidate.

    Returns:
        A list containing nine candidate actions.
    """

    if not 0.0 < translation_magnitude <= 1.0:
        raise ValueError("translation_magnitude must be in (0, 1].")

    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive.")

    magnitude = float(translation_magnitude)

    candidates = [
        CandidateAction(
            action_id="move_x_positive",
            name="Move positive X",
            description="Move along positive X",
            vector=(magnitude, 0, 0, 0, 0, 0, 0),
            horizon_steps=horizon_steps,
        ),
        CandidateAction(
            action_id="move_x_negative",
            name="Move negative X",
            description="Move along negative X",
            vector=(-magnitude, 0, 0, 0, 0, 0, 0),
            horizon_steps=horizon_steps,
        ),
        CandidateAction(
            action_id="move_y_positive",
            name="Move positive Y",
            description="Move along positive Y",
            vector=(0, magnitude, 0, 0, 0, 0, 0),
            horizon_steps=horizon_steps,
        ),
        CandidateAction(
            action_id="move_y_negative",
            name="Move negative Y",
            description="Move along negative Y",
            vector=(0, -magnitude, 0, 0, 0, 0, 0),
            horizon_steps=horizon_steps,
        ),
        CandidateAction(
            action_id="move_z_positive",
            name="Move positive Z",
            description="Move upward",
            vector=(0, 0, magnitude, 0, 0, 0, 0),
            horizon_steps=horizon_steps,
        ),
        CandidateAction(
            action_id="move_z_negative",
            name="Move negative Z",
            description="Move downward",
            vector=(0, 0, -magnitude, 0, 0, 0, 0),
            horizon_steps=horizon_steps,
        ),
        CandidateAction(
            action_id="open_gripper",
            name="Open gripper",
            description="Command gripper to open",
            vector=(0, 0, 0, 0, 0, 0, -1),
            horizon_steps=horizon_steps,
        ),
        CandidateAction(
            action_id="close_gripper",
            name="Close gripper",
            description="Command gripper to close",
            vector=(0, 0, 0, 0, 0, 0, 1),
            horizon_steps=horizon_steps,
        ),
        CandidateAction(
            action_id="hold_position",
            name="Hold position",
            description="Maintain the current command",
            vector=(0, 0, 0, 0, 0, 0, 0),
            horizon_steps=horizon_steps,
        ),
    ]

    return candidates


if __name__ == "__main__":
    candidates = generate_candidate_actions()

    print("=" * 65)
    print("CANDIDATE ACTION GENERATOR")
    print("=" * 65)

    for candidate in candidates:
        print(f"\nID: {candidate.action_id}")
        print(f"Name: {candidate.name}")
        print(f"Description: {candidate.description}")
        print(f"Action vector: {candidate.to_numpy()}")
        print(f"Horizon: {candidate.horizon_steps} steps")

    print("\nTotal candidates:", len(candidates))
