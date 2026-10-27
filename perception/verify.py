"""Plan-guided verification: look exactly where the plan says each object should now be.

For every expected object one SAM 3.1 point prompt (all objects in a single request) is placed on the front face
of the cube at its expected 3-D position. An object counts as present when
  - SAM finds an object at that point (confidence >= 0.5),
  - the front-face pixels there have the object's colour, and
  - those pixels changed compared with the image taken before acting (a cube's vertical face is shaded unlike the
    flat mat or table that was there before) -- this is what separates "red cube on the red mat" from "red mat".
Searching the whole image for "cube" misses cubes on mats of their own colour and sees a tower as one object.
"""
import sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apprentice import meta_client as mc
from perception.scene import colour_name
from sim.tabletop import CUBE, table_to_pixel


def expected_positions(plan, zones, start_belief):
    """3-D centre each moved cube should end at, following the plan in order (towers build up)."""
    where = dict(start_belief)
    out = {}
    for obj, kind, colour in plan:
        if kind == "zone" and colour in zones:
            x, y, _ = zones[colour]; z = CUBE
        elif kind == "cube" and colour in where:
            x, y, bz = where[colour]; z = bz + 2 * CUBE
        else:
            continue
        where[obj] = out[obj] = (x, y, z)
    return out


def verify(after_png, before_png, expected, cam, tag="verify", change_thresh=18.0):
    """-> ({obj: bool}, usd, details)."""
    if not expected:
        return {}, 0.0, {}
    after, before = imageio.imread(after_png)[..., :3].astype(float), imageio.imread(before_png)[..., :3].astype(float)
    names = list(expected)[:16]
    pts, toks = {}, ""
    for i, obj in enumerate(names):
        x, y, z = expected[obj]
        u, v = table_to_pixel(x, y - 0.8 * CUBE, z, cam)
        pts[obj] = (u, v)
        toks += f"{i + 1}<|point;x={int(round(u))};y={int(round(v))};w={cam['W']};h={cam['H']}|>"
    mc._check_budget(mc.SAM_IMAGE)
    d = mc._post("/responses", {"model": "sam-3.1", "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "<0f>" + toks}, {"type": "input_image", "image_url": mc._data_url(after_png)}]}],
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"}})
    mc._log("sam-3.1", mc.SAM_IMAGE, d.get("usage", {}), tag)
    text = "".join(c.get("text", "") for o in d.get("output", []) for c in o.get("content", []))
    recs = {r["id"]: r for r in mc.parse_sam(text)}
    ok, details = {}, {}
    for i, obj in enumerate(names):
        u, v = pts[obj]
        r = recs.get(i + 1)
        patch = (u - 3, v - 3, u + 3, v + 3)
        col = colour_name(after.astype(np.uint8), patch)
        ya, yb, xa, xb = int(v - 3), int(v + 4), int(u - 3), int(u + 4)
        change = float(np.abs(after[ya:yb, xa:xb] - before[ya:yb, xa:xb]).mean())
        found = bool(r) and r["c"] >= 0.5
        ok[obj] = found and col == obj and change > change_thresh
        details[obj] = {"point": (round(u), round(v)), "sam_c": r["c"] if r else None, "colour": col,
                        "change": round(change, 1)}
    return ok, mc.SAM_IMAGE, details
