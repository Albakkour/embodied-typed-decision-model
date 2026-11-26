"""Shared candidate-action vocabulary for S1 (simulation) and S2 (real robots).

Both streams must turn continuous end-effector motion into the same discrete
primitives so the model sees one action language. Directions are expressed in
the robot base frame: +x forward (away from the robot base), +y left, +z up.
"""
import numpy as np

MOVE = {
    "forward": np.array([1.0, 0.0, 0.0]),
    "backward": np.array([-1.0, 0.0, 0.0]),
    "left": np.array([0.0, 1.0, 0.0]),
    "right": np.array([0.0, -1.0, 0.0]),
    "up": np.array([0.0, 0.0, 1.0]),
    "down": np.array([0.0, 0.0, -1.0]),
}
OPPOSITE = {"forward": "backward", "backward": "forward", "left": "right",
            "right": "left", "up": "down", "down": "up",
            "open": "close", "close": "open"}

PRIMITIVES = list(MOVE) + ["open", "close"]

# Appended to every robot-action question so the model never has to guess the
# frame. Camera placement differs between datasets (and between Bridge scenes),
# so directions are always named in the robot frame, never in image space.
FRAME_NOTE = ("Directions are from the robot's point of view: forward is away from "
              "the robot base, left is the robot's left.")


def primitive_text(p):
    """Natural-language option text for a primitive."""
    if p in MOVE:
        return f"move the gripper {p}"
    return f"{p} the gripper"


def classify_motion(delta, min_dist, min_ratio):
    """Map a 3D displacement to its dominant direction.

    Returns (primitive, per_direction_projection) or (None, proj) when the
    motion is too small or not clearly dominated by one axis.
    """
    proj = {name: float(delta @ vec) for name, vec in MOVE.items()}
    mags = np.abs(delta)
    order = np.argsort(mags)[::-1]
    if mags[order[0]] < min_dist:
        return None, proj
    if mags[order[1]] > 0 and mags[order[0]] / mags[order[1]] < min_ratio:
        return None, proj
    best = max(proj, key=proj.get)
    return best, proj


def wrong_candidates(answer, proj, gripper_closed, still_dist=0.01):
    """All primitives that are clearly wrong but still meaningful in this state.

    - A direction is wrong if the gripper did not move along it
      (projection < still_dist). The opposite direction always qualifies.
    - The gripper action that would change the current gripper state is wrong
      when the executed action was a motion.
    - No-op gripper actions ("open" while open, "close" while closed) are never
      offered: they can be rejected from the text alone, without the image.
    """
    out = [p for p in MOVE if p != answer and proj.get(p, 0.0) < still_dist]
    if answer in MOVE:
        out.append("open" if gripper_closed else "close")
    return out


class DistractorSampler:
    """Pick distractors that leak as little as possible through the text alone.

    Two shortcuts are removed:
    1. Opposite-pair cue. The answer's opposite direction is almost always a
       valid distractor, but other opposite pairs rarely are, so "the answer is
       in the opposite pair" would be a strong blind heuristic. Every question
       therefore contains exactly one opposite pair: half of the time it is
       (answer, opposite), half of the time it is two distractors.
    2. Class-prior cue. Each candidate is weighted by
       (times it was the answer + 1) / (times it was a distractor + 1), updated
       online, so every primitive appears as a distractor about as often as it
       is the correct answer.
    """

    def __init__(self, rng, k=3, still_dist=0.01):
        assert k == 3, "the pair structure below assumes 4 options"
        self.rng, self.k, self.still_dist = rng, k, still_dist
        self.as_answer = {p: 0 for p in PRIMITIVES}
        self.as_distractor = {p: 0 for p in PRIMITIVES}

    def _weight(self, p):
        return (self.as_answer[p] + 1) / (self.as_distractor[p] + 1)

    def _pick(self, pool):
        w = np.array([self._weight(p) for p in pool])
        return pool[self.rng.choice(len(pool), p=w / w.sum())]

    def _singles(self, pool, taken, n):
        """Add n primitives from pool that form no opposite pair with anything taken."""
        out = []
        for _ in range(n):
            ok = [p for p in pool if p not in taken + out
                  and OPPOSITE[p] not in taken + out]
            if not ok:
                return None
            out.append(self._pick(ok))
        return out

    def __call__(self, answer, proj, gripper_closed, pool=None):
        """Return 3 distractors, or None when the pair structure is impossible.

        `pool` overrides the heuristic set of wrong candidates, e.g. with the
        candidates that S1 verified as wrong in simulation.
        """
        if pool is None:
            pool = wrong_candidates(answer, proj, gripper_closed, self.still_dist)
        answer_pair_ok = answer in MOVE and OPPOSITE[answer] in pool
        other_pairs = [p for p in pool if p in MOVE and OPPOSITE[p] in pool
                       and p not in (answer, OPPOSITE[answer])]
        use_answer_pair = answer_pair_ok and (not other_pairs or self.rng.random() < 0.5)
        if use_answer_pair:
            rest = [p for p in pool if p != OPPOSITE[answer]]
            singles = self._singles(rest, [answer, OPPOSITE[answer]], 2)
            out = None if singles is None else [OPPOSITE[answer]] + singles
        elif other_pairs:
            x = self._pick(other_pairs)
            rest = [p for p in pool if p not in (x, OPPOSITE[x], OPPOSITE.get(answer))]
            singles = self._singles(rest, [answer, x, OPPOSITE[x]], 1)
            out = None if singles is None else [x, OPPOSITE[x]] + singles
        else:
            out = None
        if out is None:
            return None
        self.as_answer[answer] += 1
        for p in out:
            self.as_distractor[p] += 1
        return out
