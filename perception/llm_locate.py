"""Baseline perception without SAM: ask a multimodal Muse model for the pixel positions of cubes and mats in a still,
then back-project with the same camera calibration. Measures what SAM 3.1 grounding adds to physical action."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apprentice import meta_client as mc
from sim.tabletop import CUBE, pixel_to_table

PROMPT = """The image (640x480 pixels, origin top-left) shows coloured cubes (4 cm) and flat coloured square mats (9 cm)
on a table. Colours are among: red, green, blue, yellow, purple, orange.
Return JSON only: {"objects": [{"kind": "cube" or "mat", "colour": "...", "u": x_pixel, "v": y_pixel}]}
where (u, v) is the centre of the object's visible bottom edge for cubes and the centre of the mat for mats."""


def llm_perceive(image_path, cam, model="muse-spark-1.3", effort="low", tag="llm_locate"):
    content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": mc._data_url(image_path)}}]
    text, usd = mc.chat([{"role": "user", "content": content}], model=model, effort=effort, max_tokens=4000, tag=tag)
    m = re.search(r"\{.*\}", text or "", re.S)
    objs = []
    if m:
        for o in json.loads(m.group(0)).get("objects", []):
            kind = "cube" if o.get("kind") == "cube" else "zone"
            if kind == "cube":  # bottom edge -> base plane, then half a cube away from the camera (as in scene.py)
                x, y = pixel_to_table(float(o["u"]), float(o["v"]), cam, 0.0); y += CUBE
                objs.append({"kind": "cube", "colour": o["colour"], "xy": (x, y), "z": CUBE})
            else:
                objs.append({"kind": "zone", "colour": o["colour"],
                             "xy": pixel_to_table(float(o["u"]), float(o["v"]), cam, 0.0), "z": 0.0})
    return objs, usd
