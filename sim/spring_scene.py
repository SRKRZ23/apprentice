"""Negative controls for Measure: a ball bobbing on a vertical spring (not free fall). Same camera and output format
as drop_scene.py; ground truth carries g=None so the evaluation expects a rejection.

  python sim/spring_scene.py data/neg/s000 --k 30 --seed 0
"""
import argparse, json, math
import mujoco, imageio.v2 as imageio

XML = """
<mujoco>
  <option timestep="{dt}" gravity="0 0 -9.81"/>
  <visual><global offwidth="{W}" offheight="{H}"/></visual>
  <worldbody>
    <light pos="-3 0 4" dir="1 0 -1"/>
    <geom type="plane" size="5 5 .1" rgba=".85 .85 .85 1"/>
    <geom type="box" pos="1.5 0 1" size=".01 3 2" rgba=".95 .95 .92 1"/>
    <body name="ball" pos="0 {x0} {z0}">
      <joint name="spring" type="slide" axis="0 0 1" stiffness="{k}" springref="{ref}" damping="{c}"/>
      <geom type="sphere" size="{r}" rgba=".9 .2 .1 1" mass=".2"/>
    </body>
    <camera name="side" pos="-{depth} 0 {cam_z}" xyaxes="0 -1 0 0 0 1" fovy="{fovy}"/>
  </worldbody>
</mujoco>
"""


def render(out, k=30.0, c=0.05, z0=1.2, x0=0.0, amp=0.25, r=0.05, fps=60, seconds=1.0, W=640, H=480,
           depth=3.0, cam_z=0.9, fovy=45.0):
    substeps = 40
    m = mujoco.MjModel.from_xml_string(XML.format(dt=1.0 / (fps * substeps), W=W, H=H, x0=x0, z0=z0, r=r, k=k, c=c,
                                                  ref=-0.2 * 9.81 / k, depth=depth, cam_z=cam_z, fovy=fovy))
    d = mujoco.MjData(m)
    d.qpos[0] = amp  # start displaced upward, released at rest
    ren = mujoco.Renderer(m, H, W)
    frames, z = [], []
    for _ in range(round(seconds * fps)):
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera="side")
        frames.append(ren.render())
        z.append(float(d.body("ball").xpos[2]))
        for _ in range(substeps):
            mujoco.mj_step(m, d)
    imageio.mimsave(out + ".mp4", frames, fps=fps, codec="libx264", quality=9, macro_block_size=8)
    f = (H / 2) / math.tan(math.radians(fovy) / 2)
    gt = {"g": None, "kind": "spring", "k": k, "fps": fps, "W": W, "H": H, "f_px": f, "depth": depth, "cam_z": cam_z,
          "px_per_m": f / depth, "radius": r, "z_true": z, "contact_frame": None}
    json.dump(gt, open(out + ".json", "w"))
    return gt


if __name__ == "__main__":
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from pathlib import Path
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for i in range(a.n):
        render(f"{a.out_dir}/s{i:03d}", k=float(rng.uniform(4, 60)), amp=float(rng.uniform(0.1, 0.35)),
               x0=float(rng.uniform(-0.8, 0.8)))
    print(f"rendered {a.n} spring clips")
