from pathlib import Path

import numpy as np
import torch
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.data.schema import CandidatePreview
from src.actions.candidate_generator import generate_candidate_actions


ROBOT_STATE_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_proprio-state",
    "object-state",
]


def extract_robot_state(obs):
    result = {}

    for key in ROBOT_STATE_KEYS:
        if key not in obs:
            continue

        value = np.asarray(obs[key])

        if np.issubdtype(value.dtype, np.number):
            result[key] = value.astype(float).tolist()

    return result


def extract_object_positions(obs):
    result = {}

    for key, value in obs.items():
        if (
            key.endswith("_pos")
            and not key.startswith("robot")
            and "_to_robot" not in key
        ):
            array = np.asarray(value)

            if array.shape == (3,):
                result[key] = array.astype(float).tolist()

    return result


def extract_goal_predicates(env):
    task_env = env.env

    results = []

    for index, goal in enumerate(task_env.parsed_problem["goal_state"]):
        predicate, subject, target = goal

        results.append(
            {
                "index": index,
                "predicate": predicate,
                "subject": subject,
                "target": target,
                "satisfied": bool(task_env._eval_predicate(goal)),
            }
        )

    return results


def synchronize_controller(env):
    controller = env.robots[0].controller

    controller.update(force=True)

    controller.set_goal(
        np.zeros(controller.control_dim),
        set_pos=controller.ee_pos.copy(),
        set_ori=controller.ee_ori_mat.copy(),
    )


def save_image(obs, path):
    image = np.asarray(obs["agentview_image"])

    Image.fromarray(image.astype(np.uint8)).save(path)


def run_candidate_preview(
    bddl_file,
    initial_state,
    candidate,
    output_dir,
    seed=0,
    camera_size=128,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action = candidate.to_numpy().astype(np.float64)

    if action.shape != (7,):
        raise ValueError("Candidate must have seven dimensions.")

    if not np.all(np.isfinite(action)):
        raise ValueError("Candidate contains non-finite values.")

    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError("Candidate exceeds action limits.")

    env = None

    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=camera_size,
            camera_widths=camera_size,
        )

        env.seed(seed)
        env.reset()

        # Restore the same initial state for every candidate.
        obs = env.set_init_state(initial_state.copy())

        # Synchronize controller position and orientation.
        synchronize_controller(env)

        robot_before = extract_robot_state(obs)

        eef_before = np.asarray(
            obs["robot0_eef_pos"],
            dtype=np.float64,
        ).copy()

        gripper_before = np.asarray(
            obs["robot0_gripper_qpos"],
            dtype=np.float64,
        ).copy()

        objects_before = extract_object_positions(obs)
        goals_before = extract_goal_predicates(env)

        success_before = bool(env.check_success())

        image_before = output_dir / f"{candidate.action_id}_before.png"

        save_image(obs, image_before)

        reward_sum = 0.0
        done = False
        executed_steps = 0

        # Execute the same candidate for its full horizon.
        for _ in range(candidate.horizon_steps):
            obs, reward, done, info = env.step(action)

            reward_sum += float(reward)
            executed_steps += 1

            if done:
                break

        eef_after = np.asarray(
            obs["robot0_eef_pos"],
            dtype=np.float64,
        ).copy()

        gripper_after = np.asarray(
            obs["robot0_gripper_qpos"],
            dtype=np.float64,
        ).copy()

        robot_after = extract_robot_state(obs)

        objects_after = extract_object_positions(obs)
        goals_after = extract_goal_predicates(env)

        success_after = bool(env.check_success())

        image_after = output_dir / f"{candidate.action_id}_after.png"

        save_image(obs, image_after)

        return CandidatePreview(
            candidate_id=candidate.action_id,
            candidate_name=candidate.name,
            action=action.tolist(),
            requested_steps=candidate.horizon_steps,
            executed_steps=executed_steps,
            eef_position_before=eef_before.tolist(),
            eef_position_after=eef_after.tolist(),
            gripper_qpos_before=gripper_before.tolist(),
            gripper_qpos_after=gripper_after.tolist(),
            robot_state_before=robot_before,
            robot_state_after=robot_after,
            object_positions_before=objects_before,
            object_positions_after=objects_after,
            goal_predicates_before=goals_before,
            goal_predicates_after=goals_after,
            task_success_before=success_before,
            task_success_after=success_after,
            reward_sum=reward_sum,
            terminated=bool(done),
            image_before=str(image_before),
            image_after=str(image_after),
        )

    finally:
        if env is not None:
            env.close()


def run_candidate_previews(
    task_id=0,
    initial_state_id=0,
    candidates=None,
    output_dir="data/debug/physics_previews",
    suite_name="libero_10",
    seed=0,
):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)

    bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )

    init_file = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )

    states = torch.load(
        init_file,
        map_location="cpu",
        weights_only=False,
    )

    initial_state = states[initial_state_id].copy()

    if candidates is None:
        candidates = generate_candidate_actions(
            translation_magnitude=0.25,
            horizon_steps=8,
        )

    results = []

    for candidate in candidates:
        print(f"Evaluating {candidate.action_id}...")

        result = run_candidate_preview(
            bddl_file=bddl_file,
            initial_state=initial_state,
            candidate=candidate,
            output_dir=output_dir,
            seed=seed,
        )

        results.append(result)

        print(
            "  Task success:",
            result.task_success_after,
        )

    return results
