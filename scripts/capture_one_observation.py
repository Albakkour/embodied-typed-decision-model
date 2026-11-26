import json
from pathlib import Path

import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# Configuration
BENCHMARK_NAME = "libero_10"
TASK_ID = 0
INITIAL_STATE_ID = 0

IMAGE_SIZE = 128
OUTPUT_DIR = Path("decision_dataset/observations")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load LIBERO task
suite = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
task = suite.get_task(TASK_ID)

bddl_file = str(
    Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
)

initial_states = suite.get_task_init_states(TASK_ID)

print("Task:", task.language)
print("Available initial states:", len(initial_states))

# 2. Create simulation environment
env = OffScreenRenderEnv(
    bddl_file_name=bddl_file,
    camera_heights=IMAGE_SIZE,
    camera_widths=IMAGE_SIZE,
)

try:
    env.seed(0)
    env.reset()

    # 3. Load one initial state
    obs = env.set_init_state(initial_states[INITIAL_STATE_ID])

    # Let the simulation settle
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)

    # 4. Capture camera image
    image = obs["agentview_image"]

    # LIBERO images are vertically flipped
    image = np.asarray(image[::-1], dtype=np.uint8)

    image_path = OUTPUT_DIR / "episode_000_step_000.png"
    Image.fromarray(image).save(image_path)

    # 5. Capture robot state
    state_keys = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
    ]

    robot_state = {}

    for key in state_keys:
        if key in obs:
            robot_state[key] = np.asarray(obs[key]).tolist()

    # 6. Save observation metadata
    record = {
        "task_id": TASK_ID,
        "task_name": task.name,
        "task_instruction": task.language,
        "initial_state_id": INITIAL_STATE_ID,
        "episode_id": 0,
        "timestep": 0,
        "image_path": str(image_path),
        "robot_state": robot_state,
    }

    metadata_path = OUTPUT_DIR / "observations.jsonl"

    with open(metadata_path, "w") as f:
        f.write(json.dumps(record) + "\n")

    print("\nObservation captured successfully!")
    print("Image:", image_path)
    print("Metadata:", metadata_path)
    print("Available observation keys:", list(obs.keys()))
    print("Robot state:", robot_state)

finally:
    env.close()
