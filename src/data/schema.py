from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CandidatePreview:
    """Recorded result of executing one candidate in simulation."""

    candidate_id: str
    candidate_name: str
    action: list[float]

    requested_steps: int
    executed_steps: int

    eef_position_before: list[float]
    eef_position_after: list[float]

    gripper_qpos_before: list[float]
    gripper_qpos_after: list[float]

    robot_state_before: dict[str, Any]
    robot_state_after: dict[str, Any]

    object_positions_before: dict[str, list[float]]
    object_positions_after: dict[str, list[float]]

    goal_predicates_before: list[dict[str, Any]]
    goal_predicates_after: list[dict[str, Any]]

    task_success_before: bool
    task_success_after: bool

    reward_sum: float
    terminated: bool

    image_before: str
    image_after: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecisionSample:
    """One supervised candidate-selection training example."""

    schema_version: str
    sample_id: str

    task: dict[str, Any]

    input: dict[str, Any]

    label: dict[str, Any]

    teacher_outcomes: list[dict[str, Any]]

    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)
