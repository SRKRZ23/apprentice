"""Coach: compare a trainee's attempt (a video) with the learned task program and name each mistake.

Mistake types: wrong_target (object ended somewhere else), missed (object never placed), wrong_order (a sequence
step done before the step it depends on), extra (an object moved that the task does not involve).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.tasks import resolve


def assess(program, obs):
    """program: learned task program. obs: understand.observe.observe() on the trainee's video."""
    cubes = {c: None for c in obs["scene"]["cubes_at_start"]}
    zones = {z: None for z in obs["scene"]["mats_at_start"]}
    plan = resolve(program, cubes, zones)
    rel = obs["final_relations"]
    moved = [e["object"].split()[0] for e in obs["events"]]
    feedback = []
    for i, (obj, kind, colour) in enumerate(plan):
        r = rel.get(obj, {"kind": "table"})
        if r.get("kind") == kind and r.get("colour") == colour:
            continue
        if obj not in moved and r.get("kind") == "table":
            feedback.append({"type": "missed", "step": i + 1, "object": obj,
                             "say": f"Step {i + 1}: the {obj} cube was not moved - it goes {_where(kind, colour)}."})
        else:
            feedback.append({"type": "wrong_target", "step": i + 1, "object": obj,
                             "say": f"Step {i + 1}: the {obj} cube is {_where(r.get('kind'), r.get('colour'))}, "
                                    f"it should be {_where(kind, colour)}."})
    if "steps" in program:  # order matters for literal sequences (stacking)
        want = [s[0] for s in plan]
        done = [m for m in moved if m in want]
        for a_i, a in enumerate(want):
            for b in want[a_i + 1:]:
                if a in done and b in done and done.index(b) < done.index(a):
                    feedback.append({"type": "wrong_order", "object": b,
                                     "say": f"The {b} cube was placed before the {a} cube - do {a} first."})
    involved = {s[0] for s in plan}
    for m in moved:
        if m not in involved:
            feedback.append({"type": "extra", "object": m, "say": f"The {m} cube is not part of this task - leave it."})
    return {"ok": not feedback, "feedback": feedback, "plan": plan}


def _where(kind, colour):
    return {"zone": f"on the {colour} mat", "cube": f"on top of the {colour} cube"}.get(kind, "on the table")
