"""Shared S1/S2 contract: what one robot-action decision means and how it is written.

Both streams import this file instead of copying values, so the model sees one
action language, one state layout and one question wording.

    S1 (simulation): candidates are executed in the simulator; the measured
        end-effector displacement and the outcome label each candidate.
    S2 (real robots): the executed trajectory is cut into windows of
        DECISION_SECONDS; the dominant displacement is the correct candidate.

Built on top of action_vocab.py (primitives, frame, distractor sampling).
"""
import hashlib
import math

import numpy as np

from .action_vocab import FRAME_NOTE, MOVE, PRIMITIVES, classify_motion, primitive_text

SCHEMA_VERSION = "1.0"

# --- What one decision covers ---------------------------------------------
# One candidate lasts DECISION_SECONDS in both streams. S1 executes it for
# round(DECISION_SECONDS * control_freq) steps; S2 aggregates the same time
# span of the recorded trajectory. To be confirmed with S1 (0.4 s vs 0.6 s).
DECISION_SECONDS = 0.6
MIN_DISPLACEMENT_M = 0.02  # below this a motion counts as "no clear motion"
MIN_AXIS_RATIO = 1.5  # dominant axis must exceed the second one by this factor
IMG_SIZE = 384  # longest image side stored in the JSONL output


def steps_per_decision(control_freq_hz):
    """Number of control/recorded steps that span one decision."""
    return max(1, round(DECISION_SECONDS * control_freq_hz))


# --- Candidate actions ----------------------------------------------------
# S1 action ids -> shared primitives. Axes are the robot base frame used by the
# LIBERO OSC controller: +x forward, +y left, +z up.
S1_ACTION_IDS = {
    "move_x_positive": "forward",
    "move_x_negative": "backward",
    "move_y_positive": "left",
    "move_y_negative": "right",
    "move_z_positive": "up",
    "move_z_negative": "down",
    "open_gripper": "open",
    "close_gripper": "close",
    "hold_position": None,  # not in the shared vocabulary yet (pending decision)
}

LIBERO_MOVE_MAGNITUDE = 0.25  # normalized OSC delta used by S1


def libero_action(primitive, gripper_command, magnitude=LIBERO_MOVE_MAGNITUDE):
    """7-D LIBERO action for a primitive.

    gripper_command is the last gripper command (-1 open, +1 close); move
    primitives repeat it so the gripper keeps its state.
    """
    a = np.zeros(7)
    if primitive in MOVE:
        a[:3] = MOVE[primitive] * magnitude
        a[6] = gripper_command
    elif primitive == "open":
        a[6] = -1.0
    elif primitive == "close":
        a[6] = 1.0
    else:
        raise ValueError(f"unknown primitive: {primitive}")
    return a


# --- Robot state ----------------------------------------------------------
# Canonical state, identical layout in every source:
#   [x, y, z, roll, pitch, yaw, gripper]
# position in metres, Euler angles in radians (XYZ convention),
# gripper closedness 0 = fully open, 1 = fully closed.
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
PANDA_MAX_FINGER_WIDTH = 0.0814  # sum of both finger joints when fully open (LIBERO demos)
GRIPPER_CLOSED_THRESHOLD = 0.5


def quat_xyzw_to_euler(q):
    """Quaternion in robosuite order [x, y, z, w] -> (roll, pitch, yaw)."""
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def canonical_state(xyz, rpy, closedness):
    """Assemble the canonical 7-D state from its parts."""
    return [round(float(v), 4) for v in (*xyz, *rpy, closedness)]


def state_from_robosuite(robot_state):
    """Canonical state from a robosuite/LIBERO observation dict (S1 sample format)."""
    qpos = robot_state["robot0_gripper_qpos"]
    width = abs(qpos[0]) + abs(qpos[1])
    closedness = min(1.0, max(0.0, 1.0 - width / PANDA_MAX_FINGER_WIDTH))
    rpy = quat_xyzw_to_euler(robot_state["robot0_eef_quat"])
    return canonical_state(robot_state["robot0_eef_pos"], rpy, closedness)


def flatten_raw_state(robot_state):
    """Full raw state as (values, keys), kept next to the canonical state."""
    values, keys = [], []
    for name, vec in robot_state.items():
        vec = np.atleast_1d(vec)
        values += [round(float(v), 5) for v in vec]
        keys += [f"{name}_{i}" for i in range(len(vec))]
    return values, keys


def gripper_text(closedness):
    return "closed" if closedness >= GRIPPER_CLOSED_THRESHOLD else "open"


# --- Displacement ---------------------------------------------------------
def displacement(eef_pos_before, eef_pos_after):
    """End-effector displacement in metres (after - before), robot base frame."""
    return np.asarray(eef_pos_after, float) - np.asarray(eef_pos_before, float)


def effective_primitive(delta):
    """Primitive that a measured displacement actually corresponds to, or None.

    S2 uses it to label windows. S1 uses it to check that a move candidate
    really moved the gripper as commanded (it can be blocked by the table or
    by an object).
    """
    prim, _ = classify_motion(np.asarray(delta, float), MIN_DISPLACEMENT_M, MIN_AXIS_RATIO)
    return prim


# --- Question text --------------------------------------------------------
def normalize_instruction(text):
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def context_text(instruction, closedness):
    return (f"Task: {normalize_instruction(instruction)} "
            f"The gripper is {gripper_text(closedness)}. {FRAME_NOTE}")


def choice_question(instruction, closedness):
    return f"{context_text(instruction, closedness)} Which action should the robot take next?"


def noul_question(instruction, closedness, primitive):
    return (f"{context_text(instruction, closedness)} "
            f"Should the robot {primitive_text(primitive)} now?")


def score_question(instruction, closedness, primitive):
    return (f"{context_text(instruction, closedness)} How much would the task advance if the robot "
            f"were to {primitive_text(primitive)} now, from 0 (not at all or worse) to 1 (a lot)?")


# --- Splits ---------------------------------------------------------------
def scene_split(key, test_buckets=1, n_buckets=10):
    """Deterministic train/test split by scene or episode key (never by sample)."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % n_buckets
    return "test" if h < test_buckets else "train"


# --- Output record ---------------------------------------------------------
REQUIRED_FIELDS = ("source", "split", "id", "image", "state", "qtype",
                   "question", "options", "answer")


def make_record(source, split, rec_id, image, state, qtype, question, options, answer, **extra):
    """One JSONL record in the shared schema; extra stream-specific fields allowed."""
    if qtype not in ("choice", "noul", "score"):
        raise ValueError(f"unknown qtype: {qtype}")
    if qtype == "choice" and answer not in options:
        raise ValueError("choice answer must be one of the options")
    rec = dict(source=source, split=split, id=rec_id, image=image, state=state,
               qtype=qtype, question=question, options=options, answer=answer)
    rec.update(extra)
    return rec


assert set(p for p in S1_ACTION_IDS.values() if p) == set(PRIMITIVES)
