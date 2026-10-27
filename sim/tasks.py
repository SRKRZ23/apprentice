"""Task families, random layouts, and scripted teacher demonstrations with ground-truth programs.

Program language (what Understand must produce and Do executes):
  {"summary": str,
   "steps": [{"object": {"colour": C}, "target": {"kind": "zone"|"cube", "colour": C}}, ...]   # literal sequence
   or
   "for_each": {"object": {"kind": "cube"}, "target": {"kind": "zone", "colour": "SAME_AS_OBJECT"}}}  # rule
"""
import numpy as np
from sim.tabletop import Tabletop, CUBE

PALETTE = ["red", "green", "blue", "yellow", "purple", "orange"]


def random_layout(rng, n_cubes, n_zones, cube_colours=None, zone_colours=None, min_gap=0.11, area=0.2):
    """Non-overlapping positions inside a square of half-width `area` in front of the parked robot."""
    pts = []
    while len(pts) < n_cubes + n_zones:
        p = rng.uniform(-area, area, 2)
        if all(np.hypot(*(p - q)) > min_gap for q in pts):
            pts.append(p)
    cc = cube_colours or list(rng.choice(PALETTE, n_cubes, replace=False))
    zc = zone_colours or list(rng.choice(PALETTE, n_zones, replace=False))
    cubes = {c: tuple(map(float, pts[i])) for i, c in enumerate(cc)}
    zones = {c: tuple(map(float, pts[n_cubes + i])) for i, c in enumerate(zc)}
    return cubes, zones


def resolve(program, cubes, zones):
    """Turn a program into concrete (object colour, target kind, target colour) steps for a given scene."""
    if "for_each" in program:
        fe = program["for_each"]
        tgt = fe["target"]
        steps = []
        for c in cubes:
            colour = c if tgt.get("colour") == "SAME_AS_OBJECT" else tgt.get("colour")
            if tgt["kind"] == "zone" and colour not in zones:
                continue
            steps.append((c, tgt["kind"], colour))
        return steps
    return [(s["object"]["colour"], s["target"]["kind"], s["target"]["colour"]) for s in program["steps"]]


def ground_truth_check(world, steps):
    """Did every step's relation end up true in the physics state? (cube on zone / cube on cube)."""
    pos = world.cube_positions()
    ok = []
    for obj, kind, colour in steps:
        p = pos.get(obj)
        if p is None:
            ok.append(False); continue
        if kind == "zone":
            zx, zy = world.zones[colour]
            ok.append(abs(p[0] - zx) < 0.045 and abs(p[1] - zy) < 0.045 and p[2] < CUBE + 0.01)
        else:
            q = pos.get(colour)
            ok.append(q is not None and np.hypot(p[0] - q[0], p[1] - q[1]) < 0.02 and 0.03 < p[2] - q[2] < 0.05)
    return ok


def teach(task, rng, record=True):
    """Scripted teacher. Returns (world, ground-truth program). task in {'sort', 'stack', 'map'}."""
    if task == "sort":   # each cube onto the zone of its own colour
        colours = list(rng.choice(PALETTE[:4], 3, replace=False))
        cubes, zones = random_layout(rng, 3, 3, colours, list(rng.permutation(colours)))
        program = {"summary": "put each cube on the mat of the same colour",
                   "for_each": {"object": {"kind": "cube"}, "target": {"kind": "zone", "colour": "SAME_AS_OBJECT"}}}
    elif task == "stack":  # build a tower in a fixed colour order on one mat
        colours = list(rng.choice(PALETTE[:4], 3, replace=False))
        zc = str(rng.choice([c for c in PALETTE if c not in colours]))
        cubes, zones = random_layout(rng, 3, 1, colours, [zc])
        program = {"summary": f"stack {colours[0]}, then {colours[1]}, then {colours[2]} on the {zc} mat",
                   "steps": [{"object": {"colour": colours[0]}, "target": {"kind": "zone", "colour": zc}},
                             {"object": {"colour": colours[1]}, "target": {"kind": "cube", "colour": colours[0]}},
                             {"object": {"colour": colours[2]}, "target": {"kind": "cube", "colour": colours[1]}}]}
    elif task == "map":  # arbitrary pairing (not same colour)
        colours = list(rng.choice(PALETTE[:3], 3, replace=False))
        zcs = list(rng.choice(PALETTE[3:], 3, replace=False))
        cubes, zones = random_layout(rng, 3, 3, colours, zcs)
        program = {"summary": "move cubes to their assigned mats",
                   "steps": [{"object": {"colour": c}, "target": {"kind": "zone", "colour": z}} for c, z in zip(colours, zcs)]}
    else:
        raise ValueError(task)
    w = Tabletop(cubes, zones, record=record)
    w.home(); w.settle(10)
    run_steps(w, resolve(program, cubes, zones), lambda c: tuple(w.cube_positions()[c]), lambda z: (*zones[z], 0.0))
    w.home(); w.settle(15)
    return w, program


def run_steps(world, steps, cube_pos, zone_pos):
    """Execute resolved steps with position lookups (ground truth for the teacher, perception for the learner)."""
    for obj, kind, colour in steps:
        sx, sy, sz = cube_pos(obj)
        if kind == "zone":
            dx, dy, _ = zone_pos(colour); dz = CUBE
        else:
            dx, dy, bz = cube_pos(colour); dz = bz + 2 * CUBE
        world.pick_place((sx, sy, sz), (dx, dy, dz), label=f"{obj}->{kind}:{colour}")
