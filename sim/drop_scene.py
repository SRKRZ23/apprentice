"""Render a free-fall clip in MuJoCo with known gravity and exact camera intrinsics.

The ball falls in the plane x = 0; the camera looks along +x from a fixed distance, so image height is an exact
linear function of world z: v = cy - f * (z - cam_z) / depth, f = (H / 2) / tan(fovy / 2).
Writes <out>.mp4 and <out>.json (ground truth: g, fps, intrinsics, true z per frame).
"""
import argparse, json, math
import numpy as np, mujoco, imageio.v2 as imageio

XML = """
<mujoco>
  <option timestep="{dt}" gravity="0 0 -{g}"/>
  <visual><global offwidth="{W}" offheight="{H}"/></visual>
  <worldbody>
    <light pos="-3 0 4" dir="1 0 -1"/>
    <geom type="plane" size="5 5 .1" rgba=".85 .85 .85 1"/>
    <geom type="box" pos="1.5 0 1" size=".01 3 2" rgba=".95 .95 .92 1"/>
    <body name="ball" pos="0 {x0} {z0}">
      <freejoint/>
      <geom type="sphere" size="{r}" rgba=".9 .2 .1 1" mass=".2"/>
    </body>
    <camera name="side" pos="-{depth} 0 {cam_z}" xyaxes="0 -1 0 0 0 1" fovy="{fovy}"/>
  </worldbody>
</mujoco>
"""


def render(out, g=9.81, z0=1.6, x0=0.0, vx=0.0, r=0.05, fps=60, seconds=1.0, W=640, H=480,
           depth=3.0, cam_z=0.9, fovy=45.0):
    substeps = 40  # integer physics steps per video frame, so frame time is exact
    m = mujoco.MjModel.from_xml_string(XML.format(dt=1.0 / (fps * substeps), g=g, W=W, H=H, x0=x0, z0=z0, r=r, depth=depth,
                                                  cam_z=cam_z, fovy=fovy))
    d = mujoco.MjData(m)
    d.qvel[0:3] = (0.0, vx, 0.0)
    ren = mujoco.Renderer(m, H, W)
    frames, z, y = [], [], []
    for _ in range(round(seconds * fps)):
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera="side")
        frames.append(ren.render())
        y.append(float(d.qpos[1])); z.append(float(d.qpos[2]))
        for _ in range(substeps):
            mujoco.mj_step(m, d)
    imageio.mimsave(out + ".mp4", frames, fps=fps, codec="libx264", quality=9, macro_block_size=8)
    f = (H / 2) / math.tan(math.radians(fovy) / 2)
    gt = {"g": g, "fps": fps, "W": W, "H": H, "f_px": f, "depth": depth, "cam_z": cam_z,
          "px_per_m": f / depth, "radius": r, "z_true": z, "y_true": y,
          "contact_frame": next((i for i, zz in enumerate(z) if zz <= r + 1e-3), None)}
    json.dump(gt, open(out + ".json", "w"))
    return gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--g", type=float, default=9.81)
    ap.add_argument("--vx", type=float, default=0.0); ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--z0", type=float, default=1.6); ap.add_argument("--seconds", type=float, default=1.0)
    a = ap.parse_args()
    gt = render(a.out, g=a.g, z0=a.z0, x0=a.x0, vx=a.vx, seconds=a.seconds)
    print(json.dumps({k: gt[k] for k in ("g", "fps", "px_per_m", "contact_frame")}))
