"""Render a randomized set of free-fall / projectile clips with ground truth.

  python scripts/gen_sim_set.py data/simset --n 50 --seed 1
g ~ U(1, 25) m/s^2 (Moon to beyond Jupiter), drop height, horizontal start and velocity; the ball stays in frame.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.drop_scene import render

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
Path(a.out).mkdir(parents=True, exist_ok=True)
params = []
for i in range(a.n):
    g = float(rng.uniform(1, 25))
    z0 = float(rng.uniform(1.2, 1.9))
    x0 = float(rng.uniform(-0.8, 0.8))
    vx = float(rng.uniform(-1.0, 1.0))
    if abs(x0 + vx) > 1.4:  # keep the ball inside the 3.3 m wide view for the whole second
        vx = -vx
    p = {"name": f"c{i:03d}", "g": g, "z0": z0, "x0": x0, "vx": vx}
    render(f"{a.out}/{p['name']}", g=g, z0=z0, x0=x0, vx=vx)
    params.append(p)
json.dump(params, open(f"{a.out}/params.json", "w"), indent=1)
print(f"rendered {a.n} clips into {a.out}")
