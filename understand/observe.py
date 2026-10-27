"""Watch a demonstration: SAM 3.1 video tracking gives the order in which objects moved; perception of the first
and last frames gives where everything started and which relations hold at the end. Output: an event list.
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apprentice.meta_client import sam
from perception.scene import perceive, colour_name


def subsample(video, fps=10):
    out = tempfile.mktemp(suffix=".mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-r", str(fps), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", out], check=True)
    return out


def still(video, which):
    """First or last frame as PNG."""
    out = tempfile.mktemp(suffix=".png")
    args = ["-sseof", "-0.1"] if which == "last" else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-i", video, "-frames:v", "1", out], check=True)
    return out


def relations(objs):
    """Final relations: cube on cube (from stack resolution) or cube inside a zone (centre within the 9 cm mat)."""
    zones = [o for o in objs if o["kind"] == "zone"]
    rel = {}
    for o in objs:
        if o["kind"] != "cube":
            continue
        if o.get("on"):
            rel[o["colour"]] = {"kind": "cube", "colour": o["on"]}
            continue
        for z in zones:
            if abs(o["xy"][0] - z["xy"][0]) < 0.05 and abs(o["xy"][1] - z["xy"][1]) < 0.05:
                rel[o["colour"]] = {"kind": "zone", "colour": z["colour"]}
        rel.setdefault(o["colour"], {"kind": "table"})
    return rel


def move_order(video10, fps=10, tag="observe"):
    """SAM 3.1 tracks every cube; a cube 'moves' when its box centre leaves its first position by > 10 px.
    Returns ([(colour, first_move_time_s)], usd)."""
    frames = imageio.mimread(video10, memtest=False)
    recs, usd = sam(video10, "cube", n_frames=len(frames), tag=f"{tag}:video")
    tracks = {}
    for r in recs:
        x1, y1, x2, y2 = r["box"]
        tracks.setdefault(r["id"], []).append((r["frame"], (x1 + x2) / 2, (y1 + y2) / 2, r["box"]))
    order = []
    for tid, tr in tracks.items():
        tr.sort()
        f0, u0, v0, box0 = tr[0]
        colour = colour_name(frames[min(f0, len(frames) - 1)], box0)
        moved = next((f for f, u, v, _ in tr if np.hypot(u - u0, v - v0) > 10), None)
        if moved is not None:
            order.append((colour, moved / fps))
    order.sort(key=lambda x: x[1])
    return order, usd


def observe(video, cam, tag="observe"):
    v10 = subsample(video)
    order, usd_v = move_order(v10, tag=tag)
    start, usd_a = perceive(still(video, "first"), cam, tag=f"{tag}:first")
    end, usd_b = perceive(still(video, "last"), cam, tag=f"{tag}:last")
    # mats are fixtures: an occupied mat is often not found again at the end, so carry the start mats over
    zones = {o["colour"]: o for o in end if o["kind"] == "zone"}
    zones.update({o["colour"]: o for o in start if o["kind"] == "zone"})
    rel = relations([o for o in end if o["kind"] == "cube"] + list(zones.values()))
    start_rel = relations(start)
    start_cubes = {o["colour"] for o in start if o["kind"] == "cube"}
    moved = []
    for c, _ in order:  # tracker id switches can report the same cube twice; gripper parts are not cubes
        if c not in moved and c in start_cubes:
            moved.append(c)
    for c, r in rel.items():  # relation changed but the tracker missed the motion: still a moved cube
        if r != start_rel.get(c) and c not in moved:
            moved.append(c)
    events = [{"step": i + 1, "object": f"{c} cube", "ended": _say(rel.get(c, {"kind": "table"}))}
              for i, c in enumerate(moved)]
    scene = {"cubes_at_start": sorted(o["colour"] for o in start if o["kind"] == "cube"),
             "mats_at_start": sorted(zones)}
    return {"scene": scene, "events": events, "final_relations": rel, "order": order,
            "usd": usd_v + usd_a + usd_b}


def _say(r):
    return {"zone": f"on the {r.get('colour')} mat", "cube": f"on top of the {r.get('colour')} cube",
            "table": "on the table"}[r["kind"]]
