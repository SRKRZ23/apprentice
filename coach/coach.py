"""Coach: compare a trainee's attempt (a video) with the learned task program and name each mistake.

Per step, plan-guided SAM 3.1 verification checks whether the object ended where the program says. For failed
steps a whole-image look says where the object went instead (or that it never moved). Video tracking gives the
order in which objects moved, which matters for sequences such as stacking.
Mistake types: wrong_target, missed, wrong_order, extra.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perception.scene import perceive
from perception.verify import expected_positions, verify
from sim.tasks import resolve
from understand.observe import subsample, still, move_order, relations


def _scene(objs):
    cubes = {o["colour"]: (o["xy"][0], o["xy"][1], o["z"]) for o in objs if o["kind"] == "cube"}
    zones = {o["colour"]: (o["xy"][0], o["xy"][1], 0.0) for o in objs if o["kind"] == "zone"}
    return cubes, zones


def coach_attempt(video, program, cam, tag="coach", fps=5):
    first, last = still(video, "first"), still(video, "last")
    start, usd = perceive(first, cam, tag=f"{tag}:first")
    cubes, zones = _scene(start)
    plan = resolve(program, cubes, zones)
    expected = expected_positions(plan, zones, cubes)
    ok, u, details = verify(last, first, expected, cam, tag=f"{tag}:verify"); usd += u
    order, u = move_order(subsample(video, fps), fps=fps, tag=f"{tag}:order"); usd += u
    feedback = []
    end, end_cubes = None, {}
    if not all(ok.get(s[0], False) for s in plan) or True:
        end, u = perceive(last, cam, tag=f"{tag}:last"); usd += u
        end_cubes, _ = _scene(end)
    zones_seen = {o["colour"]: o for o in start if o["kind"] == "zone"}
    rel = relations([o for o in end if o["kind"] == "cube"] + list(zones_seen.values())) if end else {}

    def moved_far(c):
        return c in cubes and c in end_cubes and np.hypot(end_cubes[c][0] - cubes[c][0], end_cubes[c][1] - cubes[c][1]) > 0.03

    for i, (obj, kind, colour) in enumerate(plan):
        if ok.get(obj):
            continue
        if obj in end_cubes and not moved_far(obj):
            feedback.append({"type": "missed", "step": i + 1, "object": obj,
                             "say": f"Step {i + 1}: the {obj} cube was not moved - it goes {_where(kind, colour)}."})
        else:
            r = rel.get(obj, {"kind": "unknown"})
            feedback.append({"type": "wrong_target", "step": i + 1, "object": obj,
                             "say": f"Step {i + 1}: the {obj} cube is {_where(r.get('kind'), r.get('colour'))}, "
                                    f"it should be {_where(kind, colour)}."})
    # order: only where a step depends on an earlier one (placing B on top of A needs A placed first), and only
    # for cubes that really changed place
    first_move = {}
    for c, t in order:
        first_move.setdefault(c, t)
    for later in plan:
        obj, kind, base = later
        if kind != "cube":
            continue
        if obj in first_move and base in first_move and moved_far(obj) and moved_far(base) \
                and first_move[obj] < first_move[base]:
            feedback.append({"type": "wrong_order", "object": obj,
                             "say": f"The {obj} cube was placed before the {base} cube it must stand on - "
                                    f"place the {base} cube first."})
    involved = {s[0] for s in plan}
    for c in cubes:
        if c not in involved and moved_far(c):
            feedback.append({"type": "extra", "object": c, "say": f"The {c} cube is not part of this task - leave it."})
    return {"ok": not feedback, "feedback": feedback, "plan": plan, "verify": details, "usd": usd}


def _where(kind, colour):
    return {"zone": f"on the {colour} mat", "cube": f"on top of the {colour} cube",
            "table": "on the table"}.get(kind, "somewhere else")
