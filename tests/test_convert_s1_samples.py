"""End-to-end check of scripts/convert_s1_samples.py on one synthetic sample.

The sample follows the format written by src/data/dataset_generator.py.
Run with `python tests/test_convert_s1_samples.py` or with pytest.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MOVES = {
    "move_x_positive": [0.02, 0, 0], "move_x_negative": [-0.02, 0, 0],
    "move_y_positive": [0, 0.02, 0], "move_y_negative": [0, -0.02, 0],
    "move_z_positive": [0, 0, 0.02], "move_z_negative": [0, 0, -0.02],
    "open_gripper": [0, 0, 0], "close_gripper": [0, 0, 0], "hold_position": [0, 0, 0],
}
SCORES = {"move_x_positive": 3.0, "move_x_negative": -4.0, "move_y_positive": 1.0,
          "move_y_negative": -2.0, "move_z_positive": -15.0, "move_z_negative": 16.0,
          "open_gripper": 0.0, "close_gripper": 0.0, "hold_position": 0.0}


def make_sample(image_path):
    eef = [-0.037, -0.005, 0.713]
    return {
        "schema_version": "1.0",
        "sample_id": "libero_10_task_00_state_000",
        "task": {"suite": "libero_10", "task_id": 0, "task_name": "LIVING_ROOM_SCENE2_x",
                 "instruction": "put both the alphabet soup and the tomato sauce in the basket",
                 "initial_state_id": 0, "goal_predicates": []},
        "input": {
            "image_path": str(image_path), "image_width": 128, "image_height": 128,
            "robot_state": {"robot0_eef_pos": eef, "robot0_eef_quat": [0.999, -0.012, 0.015, 0.0],
                            "robot0_gripper_qpos": [0.039, -0.039],
                            "robot0_joint_pos": [0.0, -0.18, 0.01, -2.4, 0.005, 2.24, 0.79],
                            "object-state": [0.1, 0.2, 0.3]},
            "candidate_actions": [{"action_id": a, "vector": [0] * 7, "horizon_steps": 8} for a in MOVES],
        },
        "label": {"target_choice": "move_z_negative", "confident": True, "training_eligible": True,
                  "score_margin": 13.0, "scores": SCORES,
                  "method": "baseline_corrected_approach_plus_goal_completion_v1",
                  "is_heuristic": True, "is_ground_truth": False},
        "teacher_outcomes": [
            {"candidate_id": a, "eef_position_before": eef,
             "eef_position_after": [e + d for e, d in zip(eef, MOVES[a])], "quality_score": SCORES[a]}
            for a in MOVES],
        "metadata": {"preview_duration_seconds": 0.4, "control_frequency_hz": 20},
    }


def test_convert_one_sample():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "images").mkdir()
        Image.new("RGB", (128, 128), (90, 120, 150)).save(tmp / "images" / "s.png")
        dataset = tmp / "dataset.jsonl"
        dataset.write_text(json.dumps(make_sample("images/s.png")) + "\n")
        out = tmp / "out"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "convert_s1_samples.py"),
                        str(dataset), str(out)], check=True, capture_output=True, cwd=tmp)

        split = "train" if (out / "choice_train.jsonl").exists() else "test"
        choice = [json.loads(x) for x in (out / f"choice_{split}.jsonl").read_text().splitlines()]
        noul = [json.loads(x) for x in (out / f"noul_{split}.jsonl").read_text().splitlines()]
        assert len(choice) == 1 and len(noul) == 2
        c = choice[0]
        for field in ("source", "split", "id", "image", "state", "qtype", "question", "options", "answer"):
            assert field in c, field
        assert c["answer"] == "move the gripper down"
        assert len(c["options"]) == 4 and c["answer"] in c["options"]
        # "open" is a no-op with an open gripper, so it is never offered.
        assert "open the gripper" not in c["options"]
        assert len(c["state"]) == 7 and c["state"][-1] < 0.5
        assert "object-state_0" not in c["state_keys"]  # privileged, never stored
        assert c["question"].startswith("Task: Put both the alphabet soup")
        assert sorted(n["answer"] for n in noul) == ["no", "yes"]
        assert (out / "img" / "libero_10_task_00_state_000.jpg").exists()


if __name__ == "__main__":
    test_convert_one_sample()
    print("convert_s1_samples: OK")
