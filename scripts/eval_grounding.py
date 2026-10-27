"""Same scenes, same program, same robot -- only the eyes differ: SAM 3.1 vs a multimodal LLM asked for pixel
coordinates. Reports localisation error (mm) and task success (physics ground truth).

  python scripts/eval_grounding.py --n 10 --out results/grounding.json
"""
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.tabletop import Tabletop
from sim.tasks import resolve, ground_truth_check, random_layout, PALETTE, run_steps
from perception.scene import perceive
from perception.llm_locate import llm_perceive
from act.execute import scene_from, _do
from apprentice.meta_client import spent

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10); ap.add_argument("--model", default="muse-spark-1.3")
ap.add_argument("--out", default="results/grounding.json"); ap.add_argument("--seed", type=int, default=321)
a = ap.parse_args()
prog = {"for_each": {"object": {"kind": "cube"}, "target": {"kind": "zone", "colour": "SAME_AS_OBJECT"}}}
rng = np.random.default_rng(a.seed)
rows = []
for i in range(a.n):
    n = int(rng.integers(3, 6))
    cols = list(rng.choice(PALETTE, n, replace=False))
    cubes, zones = random_layout(rng, n, n, cols, list(rng.permutation(cols)), min_gap=0.1, area=0.22)
    truth = resolve(prog, cubes, zones)
    row = {"scene": i, "n": n}
    for eyes in ("sam", "llm"):
        w = Tabletop(cubes, zones); w.home(); w.settle(5)
        png = tempfile.mktemp(suffix=".png"); imageio.imwrite(png, w.render_still())
        objs, usd = (perceive(png, w.camera(), tag=f"ground:{i}") if eyes == "sam"
                     else llm_perceive(png, w.camera(), model=a.model, tag=f"ground:{i}"))
        pc, pz = scene_from(objs)
        gt = w.cube_positions()
        errs = [1000 * np.hypot(pc[c][0] - gt[c][0], pc[c][1] - gt[c][1]) for c in pc if c in gt]
        errs += [1000 * np.hypot(pz[c][0] - zones[c][0], pz[c][1] - zones[c][1]) for c in pz if c in zones]
        found = sum(c in pc for c in cubes) + sum(c in pz for c in zones)
        _do(w, resolve(prog, pc, pz), dict(pc), pz)
        ok = ground_truth_check(w, truth)
        row[eyes] = {"success": all(ok), "steps_ok": int(sum(ok)), "steps": len(ok), "found": found,
                     "objects": 2 * n, "median_err_mm": float(np.median(errs)) if errs else None,
                     "max_err_mm": float(np.max(errs)) if errs else None, "usd": usd}
    rows.append(row)
    print(json.dumps(row))
summary = {}
for eyes in ("sam", "llm"):
    rs = [r[eyes] for r in rows]
    summary[eyes] = {"task_success": sum(r["success"] for r in rs) / len(rs),
                     "step_success": sum(r["steps_ok"] for r in rs) / sum(r["steps"] for r in rs),
                     "objects_found": sum(r["found"] for r in rs) / sum(r["objects"] for r in rs),
                     "median_err_mm": float(np.median([r["median_err_mm"] for r in rs if r["median_err_mm"] is not None])),
                     "usd_per_scene": float(np.mean([r["usd"] for r in rs]))}
summary["model"] = a.model
json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
print(json.dumps(summary, indent=1)); print(f"ledger ${spent():.3f}")
