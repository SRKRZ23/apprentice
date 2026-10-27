"""Scene perception from one image: SAM 3.1 finds the objects, colour is read locally from the pixels inside each
box, and boxes are back-projected to table coordinates with the calibrated camera. Stacks are resolved geometrically.
"""
import colorsys, sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apprentice.meta_client import sam
from sim.tabletop import CUBE, COLOURS, pixel_to_table

_REF = {c: colorsys.rgb_to_hsv(*[float(x) for x in rgb.split()]) for c, rgb in COLOURS.items()}


def colour_name(img, box, region="centre"):
    """Median colour of part of the box -> nearest named colour (hue for saturated, value for grey).

    region="centre": inner 40% (cubes). region="front": the strip along the box's bottom edge, i.e. the front of a
    mat, which stays visible when a cube of another colour sits in the middle of it.
    """
    x1, y1, x2, y2 = box
    cx, hw = (x1 + x2) / 2, max((x2 - x1) * 0.2, 1)
    if region == "front":
        ya, yb = y2 - 0.22 * (y2 - y1), y2 - 0.06 * (y2 - y1)
    else:
        cy, hh = (y1 + y2) / 2, max((y2 - y1) * 0.2, 1)
        ya, yb = cy - hh, cy + hh
    patch = img[int(ya):int(yb) + 1, int(cx - hw):int(cx + hw) + 1, :3].reshape(-1, 3) / 255.0
    h, s, v = colorsys.rgb_to_hsv(*np.median(patch, 0))
    best, bd = None, 9.0
    for name, (rh, rs, rv) in _REF.items():
        if rs < 0.2 or s < 0.2:  # achromatic: compare brightness and saturation
            dist = abs(v - rv) + abs(s - rs)
        else:
            dh = min(abs(h - rh), 1 - abs(h - rh))
            dist = 4 * dh + 0.5 * abs(s - rs) + 0.5 * abs(v - rv)
        if dist < bd:
            best, bd = name, dist
    return best


def cube_xy(box, cam, z):
    """Cube centre from the bottom edge of its box (the front-bottom edge, never hidden by a cube stacked on top):
    back-project onto the plane of the cube's base, then step half a cube away from the camera."""
    x1, y1, x2, y2 = box
    x, y = pixel_to_table((x1 + x2) / 2, y2, cam, z - CUBE)
    return x, y + CUBE


def footprint_m(box, cam, z=0.0):
    """Width of the box in metres on the plane z (perspective-correct size, independent of distance)."""
    x1, y1, x2, y2 = box
    return abs(pixel_to_table(x2, y2, cam, z)[0] - pixel_to_table(x1, y2, cam, z)[0])


def perceive(image_path, cam, prompts=("cube", "flat square mat"), tag="perceive"):
    """-> (objects, usd). objects: [{kind, colour, box, c, xy, z}] with xy in table metres.

    Kind is decided by measured size, not by the prompt that found it: SAM's "cube" also returns the flat 9 cm mats,
    and a 4 cm cube and a 9 cm mat are unambiguous once the box is converted to metres.
    """
    img = imageio.imread(image_path)
    found, usd = [], 0.0
    for prompt in prompts:
        recs, u = sam(image_path, prompt, tag=f"{tag}:{prompt}")
        usd += u
        for r in recs:
            if not any(_iou(r["box"], f["box"]) > 0.6 for f in found):
                found.append(r)
    objs = []
    for r in found:
        kind = "cube" if footprint_m(r["box"], cam) < 0.07 else "zone"
        z = CUBE if kind == "cube" else 0.0
        x1, y1, x2, y2 = r["box"]
        xy = cube_xy(r["box"], cam, z) if kind == "cube" else pixel_to_table((x1 + x2) / 2, (y1 + y2) / 2, cam, 0.0)
        objs.append({"kind": kind, "colour": colour_name(img, r["box"], "centre" if kind == "cube" else "front"),
                     "box": r["box"], "c": r["c"], "xy": xy, "z": z})
    _resolve_stacks([o for o in objs if o["kind"] == "cube"], cam)
    return objs, usd


def _resolve_stacks(cubes, cam):
    """A cube whose box sits directly on top of another cube's box (x-overlap, touching vertically) is stacked:
    re-project it at the stacked height. Repeats for towers."""
    cubes.sort(key=lambda o: -o["box"][3])  # bottom of image first = nearest/lowest
    for _ in range(3):
        for top in cubes:
            for base in cubes:
                if top is base:
                    continue
                tx1, ty1, tx2, ty2 = top["box"]
                bx1, by1, bx2, by2 = base["box"]
                overlap = max(0, min(tx2, bx2) - max(tx1, bx1)) / max(min(tx2 - tx1, bx2 - bx1), 1)
                touching = abs(ty2 - by1) < 0.35 * (by2 - by1) or (ty1 < by1 < ty2 < by2)
                if overlap > 0.6 and touching and ty1 < by1:
                    z = base["z"] + 2 * CUBE
                    if z > top["z"]:
                        top["z"] = z
                        top["xy"] = cube_xy(top["box"], cam, z)
                        top["on"] = base["colour"]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
