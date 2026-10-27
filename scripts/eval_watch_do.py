"""Watch one demonstration per task, then do the task in N new scenes. Compares:
  replay   - replay the teacher's gripper coordinates in the new scene (no perception)
  literal  - copy exactly what was done (same colours, same targets), with perfect positions
  ours     - Understand program (SAM 3.1 + Muse Spark) executed from SAM 3.1 perception, vision-verified + 1 retry
  oracle   - true program with perfect positions (upper bound of the robot mechanics)
Success = every step of the TRUE intent holds in the physics state at the end.

  python scripts/eval_watch_do.py --n 20 --out results/watch_do.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.tabletop import Tabletop, CUBE
from sim.tasks import teach, resolve, ground_truth_check, run_steps, random_layout, PALETTE
from understand.observe import observe
from understand.program import understand
from act.execute import execute
from apprentice.meta_client import spent

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20); ap.add_argument("--tasks", default="sort,stack,map")
ap.add_argument("--out", default="results/watch_do.json"); ap.add_argument("--seed", type=int, default=100)
a = ap.parse_args()


def test_scene(task, gt_prog, demo_world, rng):
    if task == "sort":  # new layout, 3-5 cubes, colours drawn from all 6 (some never seen in the demo)
        n = int(rng.integers(3, 6))
        cols = list(rng.choice(PALETTE, n, replace=False))
        return random_layout(rng, n, n, cols, list(rng.permutation(cols)), min_gap=0.1, area=0.22)
    # same objects as the demo in a new layout, plus one distractor cube
    cols = list(demo_world.cubes)
    zcs = list(demo_world.zones)
    extra = str(rng.choice([c for c in PALETTE if c not in cols + zcs] or [c for c in PALETTE if c not in cols]))
    return random_layout(rng, len(cols) + 1, len(zcs), cols + [extra], zcs, min_gap=0.1, area=0.22)


def literal_program(obs):
    steps = []
    for e in obs["events"]:
        obj = e["object"].split()[0]
        r = obs["final_relations"].get(obj, {})
        if r.get("kind") in ("zone", "cube"):
            steps.append({"object": {"colour": obj}, "target": {"kind": r["kind"], "colour": r["colour"]}})
    return {"steps": steps}


rows, programs = [], {}
partial = Path(a.out + ".partial.jsonl")
done = {}
if partial.exists():  # resume: episodes already paid for are not re-run
    for l in partial.read_text().splitlines():
        r = json.loads(l); done[(r["task"], r["scene"])] = r
rng = np.random.default_rng(a.seed)
for task in a.tasks.split(","):
    demo, gt_prog = teach(task, np.random.default_rng(0), record=True)
    video = f"data/table/teach_{task}.mp4"
    demo.save_video(video)
    cache = Path(f"data/table/teach_{task}.learned.json")
    if cache.exists():
        c = json.loads(cache.read_text()); obs, prog, u_usd = c["obs"], c["prog"], c["understand_usd"]
    else:
        obs = observe(video, demo.camera(), tag=f"wd:{task}:observe")
        prog, u_usd, raw = understand(obs, tag=f"wd:{task}:understand")
        cache.write_text(json.dumps({"obs": obs, "prog": prog, "understand_usd": u_usd, "raw": raw}, default=str))
    programs[task] = {"truth": gt_prog, "learned": prog, "events": obs["events"],
                      "observe_usd": obs["usd"], "understand_usd": u_usd}
    print(f"== {task}: learned {json.dumps(prog)}")
    lit = literal_program(obs)
    replay = [(e["src"], e["dst"]) for e in demo.log]
    for i in range(a.n):
        cubes, zones = test_scene(task, gt_prog, demo, rng)
        truth_steps = resolve(gt_prog, cubes, zones)
        if (task, i) in done:
            rows.append(done[(task, i)]); continue
        row = {"task": task, "scene": i, "cubes": list(cubes), "zones": list(zones)}
        # replay
        w = Tabletop(cubes, zones); w.home(); w.settle(5)
        for src, dst in replay:
            w.pick_place(src, dst)
        row["replay"] = all(ground_truth_check(w, truth_steps))
        # literal copy, perfect positions
        w = Tabletop(cubes, zones); w.home(); w.settle(5)
        steps = [s for s in resolve(lit, cubes, zones) if s[0] in cubes and (s[2] in zones if s[1] == "zone" else s[2] in cubes)]
        run_steps(w, steps, lambda c: tuple(w.cube_positions()[c]), lambda z: (*zones[z], 0.0))
        row["literal"] = all(ground_truth_check(w, truth_steps))
        # oracle
        w = Tabletop(cubes, zones); w.home(); w.settle(5)
        run_steps(w, truth_steps, lambda c: tuple(w.cube_positions()[c]), lambda z: (*zones[z], 0.0))
        row["oracle"] = all(ground_truth_check(w, truth_steps))
        # ours
        t0 = time.time()
        w = Tabletop(cubes, zones)
        rep = execute(w, prog, tag=f"wd:{task}:{i}")
        gt_ok = ground_truth_check(w, truth_steps)
        row.update({"ours": all(gt_ok), "ours_vision_says": all(rep["vision_verdict"]) and len(rep["plan"]) == len(truth_steps),
                    "retries": rep["retries"], "usd": rep["usd"], "sec": round(time.time() - t0, 1),
                    "plan_matches_truth": sorted(rep["plan"]) == sorted(truth_steps)})
        rows.append(row)
        with partial.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(json.dumps({k: row[k] for k in ("task", "scene", "replay", "literal", "oracle", "ours", "ours_vision_says",
                                              "retries", "plan_matches_truth", "usd")}))

summary = {}
for task in a.tasks.split(",") + ["all"]:
    rs = [r for r in rows if task == "all" or r["task"] == task]
    summary[task] = {m: round(sum(r[m] for r in rs) / len(rs), 3) for m in ("replay", "literal", "oracle", "ours")}
    summary[task]["n"] = len(rs)
    summary[task]["vision_agrees_with_physics"] = round(sum(r["ours"] == r["ours_vision_says"] for r in rs) / len(rs), 3)
    summary[task]["false_success_claims"] = sum(r["ours_vision_says"] and not r["ours"] for r in rs)
    summary[task]["usd_per_episode"] = round(float(np.mean([r["usd"] for r in rs])), 4)
json.dump({"summary": summary, "programs": programs, "rows": rows}, open(a.out, "w"), indent=1, default=str)
print(json.dumps(summary, indent=1))
print(f"ledger ${spent():.3f}")
