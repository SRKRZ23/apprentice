"""Tabletop world: coloured cubes, optional coloured target zones, and a Cartesian (gantry) robot with a two-finger
gripper that holds cubes by friction. Fixed camera with known intrinsics so pixels map back to the table plane.

    w = Tabletop(cubes={"red": (0.1, 0.0)}, zones={"blue": (-0.1, 0.1)}, record=True)
    w.pick_place((0.1, 0.0, CUBE), (-0.1, 0.1, CUBE))
    w.save_video("out.mp4"); w.cube_positions()
"""
import math
import numpy as np, mujoco, imageio.v2 as imageio

CUBE = 0.02          # cube half-size = centre height when resting on the table
HOVER = 0.16         # finger-centre height while travelling
FPS, SUBSTEPS = 30, 16
COLOURS = {"red": ".85 .15 .1", "green": ".15 .65 .2", "blue": ".15 .3 .85", "yellow": ".95 .8 .1",
           "purple": ".55 .2 .7", "orange": ".95 .5 .1", "white": ".92 .92 .92", "black": ".12 .12 .12"}
CARRIAGE_Z0 = 0.30   # carriage rest height; finger centres sit 0.045 below the carriage
FINGER_DROP = 0.045


def scene_xml(cubes, zones, W, H):
    cube_xml = "".join(
        f'<body name="cube_{c}" pos="{x} {y} {CUBE}"><freejoint/>'
        f'<geom type="box" size="{CUBE} {CUBE} {CUBE}" rgba="{COLOURS[c]} 1" mass=".05" friction="1.5 .05 .01"/></body>'
        for c, (x, y) in cubes.items())
    zone_xml = "".join(
        f'<geom name="zone_{c}" type="box" pos="{x} {y} 0.0008" size=".045 .045 .0008" rgba="{COLOURS[c]} 1" '
        f'contype="0" conaffinity="0"/>' for c, (x, y) in zones.items())
    return f"""
<mujoco>
  <option timestep="{1 / (FPS * SUBSTEPS)}" gravity="0 0 -9.81"/>
  <visual><global offwidth="{W}" offheight="{H}"/><quality shadowsize="2048"/></visual>
  <asset><texture name="wood" type="2d" builtin="flat" rgb1=".78 .66 .5" width="64" height="64"/>
         <material name="table" texture="wood" rgba=".78 .66 .5 1"/></asset>
  <worldbody>
    <light pos="0.3 -0.4 1.2" dir="-0.2 0.3 -1" diffuse=".8 .8 .8"/>
    <light pos="-0.4 0.2 1.0" dir="0.3 -0.1 -1" diffuse=".35 .35 .35" castshadow="false"/>
    <geom type="plane" size=".6 .6 .02" material="table"/>
    {zone_xml}
    {cube_xml}
    <body name="carriage" pos="0 0 {CARRIAGE_Z0}" gravcomp="1">
      <joint name="gx" type="slide" axis="1 0 0" range="-.45 .45" damping="60"/>
      <joint name="gy" type="slide" axis="0 1 0" range="-.45 .45" damping="60"/>
      <joint name="gz" type="slide" axis="0 0 1" range="-.29 .25" damping="60"/>
      <geom type="box" size=".03 .055 .015" rgba=".3 .3 .35 1" mass=".5"/>
      <geom type="cylinder" fromto="0 0 .015 0 0 .7" size=".01" rgba=".55 .55 .6 1" contype="0" conaffinity="0" mass=".01"/>
      <body name="lf" pos="0 .035 -.045" gravcomp="1">
        <joint name="fl" type="slide" axis="0 -1 0" range="0 .03" damping="5"/>
        <geom type="box" size=".012 .006 .03" friction="2 .05 .01" rgba=".15 .15 .15 1" mass=".05"/>
      </body>
      <body name="rf" pos="0 -.035 -.045" gravcomp="1">
        <joint name="fr" type="slide" axis="0 1 0" range="0 .03" damping="5"/>
        <geom type="box" size=".012 .006 .03" friction="2 .05 .01" rgba=".15 .15 .15 1" mass=".05"/>
      </body>
    </body>
    <camera name="cam" pos="0 -.78 .78" xyaxes="1 0 0 0 .75 .8" fovy="45"/>
  </worldbody>
  <actuator>
    <position joint="gx" kp="3000" forcerange="-200 200"/>
    <position joint="gy" kp="3000" forcerange="-200 200"/>
    <position joint="gz" kp="3000" forcerange="-200 200"/>
    <position joint="fl" kp="300" forcerange="-12 12"/>
    <position joint="fr" kp="300" forcerange="-12 12"/>
  </actuator>
</mujoco>"""


class Tabletop:
    def __init__(self, cubes, zones=None, record=False, W=640, H=480):
        self.cubes, self.zones = dict(cubes), dict(zones or {})
        self.W, self.H = W, H
        self.m = mujoco.MjModel.from_xml_string(scene_xml(self.cubes, self.zones, W, H))
        self.d = mujoco.MjData(self.m)
        self.record = record
        self.frames, self.log = [], []
        self.ren = mujoco.Renderer(self.m, H, W) if record else None
        self.target = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # gx gy gz fl fr
        self.t = 0
        mujoco.mj_forward(self.m, self.d)
        self.settle(10)

    # --- low level -----------------------------------------------------------------------------------------
    def _frame(self):
        self.d.ctrl[:] = self.target
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        if self.record:
            self.ren.update_scene(self.d, camera="cam")
            self.frames.append(self.ren.render())

    def settle(self, n):
        for _ in range(n):
            self._frame()

    def _move(self, x, y, h, seconds):
        """Linear interpolation of the finger-centre position (x, y, height) over `seconds`."""
        start = self.target[:3].copy()
        goal = np.array([x, y, h + FINGER_DROP - CARRIAGE_Z0])
        n = max(int(seconds * FPS), 1)
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)  # smoothstep
            self.target[:3] = start + s * (goal - start)
            self._frame()

    def _grip(self, closed, seconds=0.4):
        self.target[3:] = 0.02 if closed else 0.0
        self.settle(int(seconds * FPS))

    # --- primitives ------------------------------------------------------------------------------------------
    def pick_place(self, src, dst, label=None):
        (sx, sy, sz), (dx, dy, dz) = src, dst
        f0 = self.t
        self._grip(False, 0.2)
        self._move(sx, sy, HOVER, 0.8)
        self._move(sx, sy, sz + 0.003, 0.5)
        self._grip(True)
        self._move(sx, sy, HOVER, 0.5)
        self._move(dx, dy, HOVER, 0.9)
        self._move(dx, dy, dz + 0.004, 0.5)
        self._grip(False)
        self._move(dx, dy, HOVER, 0.4)
        self.log.append({"action": "pick_place", "label": label, "src": src, "dst": dst, "frames": (f0, self.t)})

    def home(self):
        """Park in the back corner, out of the camera's line of sight to the work area."""
        self._move(0.38, 0.40, HOVER + 0.1, 0.7)

    # --- state / perception helpers ---------------------------------------------------------------------------
    def cube_positions(self):
        return {c: self.d.body(f"cube_{c}").xpos.copy() for c in self.cubes}

    def render_still(self):
        ren = self.ren or mujoco.Renderer(self.m, self.H, self.W)
        ren.update_scene(self.d, camera="cam")
        return ren.render()

    def save_video(self, path):
        imageio.mimsave(path, self.frames, fps=FPS, codec="libx264", quality=8, macro_block_size=8)

    def camera(self):
        """Intrinsics/extrinsics of the fixed camera (for pixel -> table back-projection)."""
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        f = (self.H / 2) / math.tan(math.radians(self.m.cam_fovy[cid]) / 2)
        return {"pos": self.d.cam_xpos[cid].tolist(), "R": self.d.cam_xmat[cid].reshape(3, 3).tolist(),
                "f": f, "cx": self.W / 2, "cy": self.H / 2, "W": self.W, "H": self.H}


def pixel_to_table(u, v, cam, height=CUBE):
    """Back-project pixel (u, v) onto the horizontal plane z = height."""
    R, pos = np.array(cam["R"]), np.array(cam["pos"])
    ray = R @ np.array([(u - cam["cx"]) / cam["f"], -(v - cam["cy"]) / cam["f"], -1.0])
    s = (height - pos[2]) / ray[2]
    p = pos + s * ray
    return float(p[0]), float(p[1])


def table_to_pixel(x, y, z, cam):
    R, pos = np.array(cam["R"]), np.array(cam["pos"])
    pc = R.T @ (np.array([x, y, z]) - pos)
    return cam["cx"] + cam["f"] * pc[0] / -pc[2], cam["cy"] - cam["f"] * pc[1] / -pc[2]
