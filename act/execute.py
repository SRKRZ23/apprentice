"""Do: execute a task program in a new scene using only perception, then verify with the same eyes.

The learner never reads simulator state: object positions come from SAM 3.1 on a camera still, and success is
judged twice -- by plan-guided SAM 3.1 verification on the final still (what the robot believes) and by physics
(ground truth, used only for scoring). Steps that verification rejects are re-planned from a fresh look and retried.
"""
import sys, tempfile
from pathlib import Path
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perception.scene import perceive
from perception.verify import expected_positions, verify
from sim.tabletop import CUBE
from sim.tasks import resolve, run_steps


def snap(world):
    path = tempfile.mktemp(suffix=".png")
    imageio.imwrite(path, world.render_still())
    return path


def scene_from(objs):
    cubes = {o["colour"]: (o["xy"][0], o["xy"][1], o["z"]) for o in objs if o["kind"] == "cube"}
    zones = {o["colour"]: (o["xy"][0], o["xy"][1], 0.0) for o in objs if o["kind"] == "zone"}
    return cubes, zones


def _feasible(step, cubes, zones):
    obj, kind, colour = step
    return obj in cubes and (colour in zones if kind == "zone" else colour in cubes)


def _do(world, steps, belief, zones):
    for step in steps:
        if not _feasible(step, belief, zones):
            continue
        run_steps(world, [step], lambda c: belief[c], lambda z: zones[z])
        obj, kind, colour = step
        x, y, _ = zones[colour] if kind == "zone" else belief[colour]
        belief[obj] = (x, y, CUBE if kind == "zone" else belief[colour][2] + 2 * CUBE)


def execute(world, program, tag="do", retry=True):
    """-> report dict with the plan, verification verdicts, retries and cost."""
    world.home(); world.settle(5)
    before = snap(world)
    cam = world.camera()
    objs, usd = perceive(before, cam, tag=f"{tag}:before")
    cubes, zones = scene_from(objs)
    plan = resolve(program, cubes, zones)
    _do(world, plan, dict(cubes), zones)
    world.home(); world.settle(10)

    expected = expected_positions(plan, zones, cubes)
    after = snap(world)
    ok, u, details = verify(after, before, expected, cam, tag=f"{tag}:verify"); usd += u
    verdict = [ok.get(s[0], False) for s in plan]
    retries = 0
    if retry and not all(verdict):
        # closed loop: find where the failed cubes really are, redo only the failed steps, verify again
        now, u = perceive(after, cam, tag=f"{tag}:relook"); usd += u
        cubes2, _ = scene_from(now)
        belief = {**{k: v for k, v in expected.items() if ok.get(k)},
                  **{k: v for k, v in cubes2.items() if not ok.get(k)}}
        failed = [s for s, v in zip(plan, verdict) if not v]
        retries = sum(_feasible(s, belief, zones) for s in failed)
        _do(world, failed, belief, zones)
        world.home(); world.settle(10)
        after2 = snap(world)
        ok, u, details = verify(after2, before, expected, cam, tag=f"{tag}:verify2"); usd += u
        verdict = [ok.get(s[0], False) for s in plan]
    return {"plan": plan, "vision_verdict": verdict, "retries": retries, "usd": usd, "verify_details": details,
            "perceived_cubes": sorted(cubes), "perceived_zones": sorted(zones)}
