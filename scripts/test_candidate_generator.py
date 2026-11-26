import numpy as np

from src.actions.candidate_generator import (
    generate_candidate_actions,
)


def main():
    candidates = generate_candidate_actions()

    assert len(candidates) == 9

    identifiers = [c.action_id for c in candidates]

    assert len(set(identifiers)) == 9

    for candidate in candidates:
        action = candidate.to_numpy()

        assert action.shape == (7,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
        assert candidate.horizon_steps == 8

        print(f"{candidate.action_id:20s} {action}")

    print("\nAll candidate-action tests passed!")


if __name__ == "__main__":
    main()
