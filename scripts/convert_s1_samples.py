"""Convert S1 LIBERO candidate-selection samples into the shared S1/S2 JSONL schema.

Input: the `dataset.jsonl` written by `scripts/generate_libero_dataset.py`
(one `DecisionSample` per line, see `src/data/schema.py`). Used fields:

  task.suite / task_id / instruction / initial_state_id
  input.image_path, input.robot_state, input.candidate_actions
  label.target_choice / training_eligible / scores / method
  teacher_outcomes[*].candidate_id / eef_position_before / eef_position_after
  metadata.preview_duration_seconds

Output (in <out_dir>), the same record format as the S2 real-robot streams:

  choice_<split>.jsonl  1 per kept sample: the teacher's target + 3 distractors
                        verified as wrong in simulation
  noul_<split>.jsonl    2 per kept sample: target -> "yes", one wrong -> "no"
  score_<split>.jsonl   only with --score-scale-mm (quality scale not agreed yet)
  img/                  observation images, upright, longest side <= 384 px
  stats.json            counts, label distribution, measured displacements

A candidate counts as wrong when its teacher score is at least --wrong-margin
below the target's score, or when it is a move that the simulation blocked
(the gripper did not move along the commanded direction).

Usage (from the repository root):
    python scripts/convert_s1_samples.py data/processed/libero_s1_pilot/dataset.jsonl \
        data/processed/libero_s1_shared [--flip vertical] [--wrong-margin 5]
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.conversion.action_vocab import MOVE, DistractorSampler  # noqa: E402
from src.conversion.shared_contract import (  # noqa: E402
    GRIPPER_CLOSED_THRESHOLD, IMG_SIZE, MIN_AXIS_RATIO, S1_ACTION_IDS, choice_question,
    displacement, flatten_raw_state, make_record, noul_question, primitive_text,
    scene_split, score_question, state_from_robosuite)

# A move is "blocked" when it advanced less than this along its own direction.
BLOCKED_MIN_M = 0.005
# Simulator-only fields that must not reach the model as state (privileged).
PRIVILEGED_STATE_KEYS = ("object-state",)


def iter_samples(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_image(image_path, dataset_dir):
    """The generator stores the path relative to its working directory."""
    p = Path(image_path)
    for cand in (p, Path.cwd() / p, dataset_dir / p, dataset_dir / "images" / p.name):
        if cand.exists():
            return cand
    return None


def no_op(primitive, gripper_closed):
    """Gripper actions that would not change anything; never offered as options."""
    return (primitive == "open" and not gripper_closed) or (primitive == "close" and gripper_closed)


def is_blocked(primitive, delta):
    """True when a move candidate did not clearly move along its commanded direction."""
    if primitive not in MOVE:
        return False
    along = float(delta @ MOVE[primitive])
    others = np.abs(delta - along * MOVE[primitive])
    return along < BLOCKED_MIN_M or along < MIN_AXIS_RATIO * float(others.max())


def save_image(src, dst, flip):
    img = Image.open(src).convert("RGB")
    if flip == "vertical":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip == "rotate180":
        img = img.rotate(180)
    img.thumbnail((IMG_SIZE, IMG_SIZE))
    img.save(dst, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", help="dataset.jsonl from generate_libero_dataset.py")
    ap.add_argument("out_dir")
    ap.add_argument("--image-prefix", default="libero")
    ap.add_argument("--flip", choices=("none", "vertical", "rotate180"), default="vertical",
                    help="physics_preview.save_image() stores raw OpenGL frames, which are "
                         "upside down; use 'none' once the generator flips them itself")
    ap.add_argument("--wrong-margin", type=float, default=5.0,
                    help="teacher-score gap (mm-based units) below the target that makes a candidate wrong")
    ap.add_argument("--score-scale-mm", type=float, default=None,
                    help="emit Score questions with answer = clip(score / scale, 0, 1)")
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="keep samples whose image file is missing (for format tests)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset).resolve().parent
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sampler = DistractorSampler(rng)
    stats, answers, targets_raw = collections.Counter(), collections.Counter(), collections.Counter()
    moved = collections.defaultdict(list)  # primitive -> displacement along its direction (m)
    preview_s = set()
    out = {}

    def write(split, qtype, rec):
        key = (split, qtype)
        if key not in out:
            out[key] = open(out_dir / f"{qtype}_{split}.jsonl", "w", encoding="utf-8")
        out[key].write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[f"{qtype}_{split}"] += 1

    for sample in iter_samples(args.dataset):
        stats["samples"] += 1
        label, inp, task = sample["label"], sample["input"], sample["task"]
        preview_s.add(sample.get("metadata", {}).get("preview_duration_seconds"))
        targets_raw[str(label.get("target_choice"))] += 1

        # Displacements of every move candidate, for calibrating the S2 window.
        before = inp["robot_state"]["robot0_eef_pos"]
        deltas = {}
        for o in sample.get("teacher_outcomes", []):
            prim = S1_ACTION_IDS.get(o["candidate_id"])
            if prim:
                deltas[prim] = displacement(o.get("eef_position_before", before), o["eef_position_after"])
                if prim in MOVE:
                    moved[prim].append(float(deltas[prim] @ MOVE[prim]))

        if not label.get("training_eligible") or label.get("target_choice") is None:
            stats["skipped_not_eligible"] += 1
            continue
        if label["target_choice"] not in S1_ACTION_IDS:
            stats["skipped_unknown_action_id"] += 1
            continue
        answer = S1_ACTION_IDS[label["target_choice"]]
        if answer is None:  # hold_position is not in the shared vocabulary
            stats["skipped_answer_not_in_vocab"] += 1
            continue

        state = state_from_robosuite(inp["robot_state"])
        closedness = state[-1]
        gripper_closed = closedness >= GRIPPER_CLOSED_THRESHOLD
        if no_op(answer, gripper_closed):
            stats["skipped_answer_is_no_op"] += 1
            continue
        if answer in deltas and is_blocked(answer, deltas[answer]):
            stats["skipped_target_blocked"] += 1
            continue

        scores = label.get("scores") or {}
        target_score = scores.get(label["target_choice"])
        offered = [S1_ACTION_IDS[c["action_id"]] for c in inp["candidate_actions"]
                   if S1_ACTION_IDS.get(c["action_id"])]
        id_of = {S1_ACTION_IDS[c["action_id"]]: c["action_id"] for c in inp["candidate_actions"]
                 if S1_ACTION_IDS.get(c["action_id"])}
        wrong = []
        for prim in offered:
            if prim == answer or no_op(prim, gripper_closed):
                continue
            sc = scores.get(id_of[prim])
            low = target_score is not None and sc is not None and sc <= target_score - args.wrong_margin
            if low or (prim in deltas and is_blocked(prim, deltas[prim])):
                wrong.append(prim)
        distractors = sampler(answer, {}, gripper_closed, pool=wrong)
        if distractors is None:
            stats["skipped_few_verified_wrong"] += 1
            continue

        src_img = resolve_image(inp["image_path"], dataset_dir)
        img_name = f"{sample['sample_id']}.jpg"
        if src_img is not None:
            save_image(src_img, img_dir / img_name, args.flip)
        elif args.allow_missing_images:
            stats["images_missing_kept"] += 1
        else:
            stats["skipped_missing_image"] += 1
            continue

        scene = (f"{task['suite']}/task{int(task['task_id']):02d}"
                 f"/init{int(task.get('initial_state_id', 0)):03d}")
        split = scene_split(scene)
        raw_state = {k: v for k, v in inp["robot_state"].items() if k not in PRIVILEGED_STATE_KEYS}
        raw, raw_keys = flatten_raw_state(raw_state)
        instr = task["instruction"]
        base = dict(source=f"S1/{task['suite']}", split=split,
                    image=f"{args.image_prefix}:img/{img_name}", state=state,
                    episode=scene, step=sample.get("step", 0),
                    state_raw=raw, state_keys=raw_keys,
                    label_method=label.get("method"), score_margin=label.get("score_margin"))
        rec_id = sample["sample_id"]

        options = [primitive_text(p) for p in [answer] + distractors]
        rng.shuffle(options)
        write(split, "choice", make_record(
            rec_id=rec_id, qtype="choice", question=choice_question(instr, closedness),
            options=options, answer=primitive_text(answer), **base))
        for prim, yes_no in ((answer, "yes"), (distractors[0], "no")):
            write(split, "noul", make_record(
                rec_id=f"{rec_id}-{prim}", qtype="noul",
                question=noul_question(instr, closedness, prim),
                options=["yes", "no"], answer=yes_no, **base))
        if args.score_scale_mm:
            for prim in offered:
                sc = scores.get(id_of[prim])
                if sc is None or no_op(prim, gripper_closed):
                    continue
                write(split, "score", make_record(
                    rec_id=f"{rec_id}-{prim}-score", qtype="score",
                    question=score_question(instr, closedness, prim), options=None,
                    answer=round(min(1.0, max(0.0, sc / args.score_scale_mm)), 4),
                    score_raw=sc, **base))
        answers[answer] += 1

    for f in out.values():
        f.close()
    report = {
        "stats": dict(stats),
        "teacher_targets_raw": dict(targets_raw),
        "answers_kept": dict(answers),
        "as_distractor": sampler.as_distractor,
        "preview_duration_seconds": sorted(p for p in preview_s if p is not None),
        "move_displacement_along_direction_m": {
            p: {"n": len(v), "median": round(float(np.median(v)), 4), "min": round(min(v), 4)}
            for p, v in moved.items()},
        "config": {"wrong_margin": args.wrong_margin, "blocked_min_m": BLOCKED_MIN_M,
                   "flip": args.flip, "score_scale_mm": args.score_scale_mm},
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
