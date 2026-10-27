"""Coach evaluation: trainee attempts with injected mistakes (and clean attempts), recorded on video, assessed
against the program learned from the teacher's demo. Reports detection precision/recall by mistake type.

  python scripts/eval_coach.py --n 30 --out results/coach.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.tabletop import Tabletop
from sim.tasks import resolve, run_steps, random_layout, PALETTE
from understand.observe import observe
from coach.coach import assess
from apprentice.meta_client import spent

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30); ap.add_argument("--programs", default="results/watch_do.json")
ap.add_argument("--out", default="results/coach.json"); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
learned = {t: p["learned"] for t, p in json.load(open(a.programs))["programs"].items()}
rng = np.random.default_rng(a.seed)
Path("data/coach").mkdir(parents=True, exist_ok=True)
KINDS = ["none", "wrong_target", "missed", "wrong_order", "extra"]

rows = []
for i in range(a.n):
    task = ["sort", "stack", "map"][i % 3]
    prog = learned[task]
    if task == "sort":
        cols = list(rng.choice(PALETTE, 3, replace=False))
        cubes, zones = random_layout(rng, 3, 3, cols, list(rng.permutation(cols)), min_gap=0.1, area=0.22)
    else:
        cols = [s["object"]["colour"] for s in prog["steps"]]
        zcs = sorted({s["target"]["colour"] for s in prog["steps"] if s["target"]["kind"] == "zone"})
        extra = str(rng.choice([c for c in PALETTE if c not in cols + zcs]))
        cubes, zones = random_layout(rng, len(cols) + 1, len(zcs), cols + [extra], zcs, min_gap=0.1, area=0.22)
    plan = resolve(prog, cubes, zones)
    kind = KINDS[int(rng.integers(len(KINDS)))]
    if kind == "wrong_order" and task != "stack":
        kind = "wrong_target"
    steps = list(plan)
    if kind == "wrong_target":   # one cube goes to another mat/cube than it should
        k = int(rng.integers(len(steps)))
        obj, tk, tc = steps[k]
        alts = [z for z in zones if z != tc] if tk == "zone" else [z for z in zones]
        steps[k] = (obj, "zone", str(rng.choice(alts)))
    elif kind == "missed":
        steps.pop(int(rng.integers(len(steps))) if task != "stack" else len(steps) - 1)
    elif kind == "wrong_order":  # stack: place the 2nd cube on the mat first, then the 1st cube on top of it
        (o1, k1, c1), (o2, _, _) = steps[0], steps[1]
        steps = [(o2, k1, c1), (o1, "cube", o2)] + steps[2:3]
        steps[2] = (steps[2][0], "cube", o1)
    elif kind == "extra":
        free = [c for c in cubes if c not in {s[0] for s in plan}]
        if free:
            used = {s[2] for s in plan if s[1] == "zone"}
            spot = [z for z in zones if z not in used] or list(zones)
            steps.append((free[0], "zone", spot[0]) if task != "sort" else (free[0], "zone", spot[0]))
        else:
            kind = "none"
    w = Tabletop(cubes, zones, record=True); w.home(); w.settle(8)
    run_steps(w, steps, lambda c: tuple(w.cube_positions()[c]), lambda z: (*zones[z], 0.0))
    w.home(); w.settle(12)
    video = f"data/coach/t{i:03d}.mp4"; w.save_video(video)
    obs = observe(video, w.camera(), tag=f"coach:{i}")
    rep = assess(prog, obs)
    found = sorted({f["type"] for f in rep["feedback"]})
    row = {"i": i, "task": task, "injected": kind, "found": found, "usd": obs["usd"],
           "feedback": [f["say"] for f in rep["feedback"]]}
    rows.append(row)
    print(json.dumps({k: row[k] for k in ("i", "task", "injected", "found", "feedback")}))

det = {k: {"tp": 0, "fn": 0, "fp": 0} for k in KINDS[1:]}
for r in rows:
    for k in KINDS[1:]:
        inj, got = r["injected"] == k, k in r["found"]
        det[k]["tp"] += inj and got; det[k]["fn"] += inj and not got; det[k]["fp"] += got and not inj
clean = [r for r in rows if r["injected"] == "none"]
summary = {"n": len(rows), "by_type": det,
           "clean_attempts": len(clean), "clean_flagged": sum(bool(r["found"]) for r in clean),
           "mistake_attempts_caught": sum(bool(r["found"]) for r in rows if r["injected"] != "none"),
           "mistake_attempts": sum(r["injected"] != "none" for r in rows),
           "exact_type_match": sum((r["injected"] == "none" and not r["found"]) or (r["injected"] in r["found"]) for r in rows),
           "usd_per_attempt": round(float(np.mean([r["usd"] for r in rows])), 4)}
json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
print(json.dumps(summary, indent=1))
print(f"ledger ${spent():.3f}")
