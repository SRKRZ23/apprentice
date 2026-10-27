"""Run SAM 3.1 on rendered drop clips and compare measured g with simulator ground truth.

  python scripts/eval_sim_drop.py data/sim/earth data/sim/moon [--prompt "red ball"] [--oracle]
--oracle skips the API and fits the true projected positions (checks the fitting code, costs nothing).
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apprentice.meta_client import sam, spent
from measure.fit import track_centres, measure

ap = argparse.ArgumentParser()
ap.add_argument("clips", nargs="+"); ap.add_argument("--prompt", default="red ball")
ap.add_argument("--oracle", action="store_true"); ap.add_argument("--cached", action="store_true")
ap.add_argument("--out", help="write rows + summary JSON here")
a = ap.parse_args()

rows = []
for clip in a.clips:
    gt = json.load(open(clip + ".json"))
    cy = gt["H"] / 2
    n = len(gt["z_true"])
    if a.cached:  # re-fit saved SAM output, no API call
        recs, usd = json.load(open(clip + ".sam.json")), 0.0
        centres = track_centres(recs)
        found = len(centres)
    elif a.oracle:
        centres = {i: (0.0, cy - gt["px_per_m"] * (z - gt["cam_z"])) for i, z in enumerate(gt["z_true"])}
        usd, found = 0.0, n
    else:
        recs, usd = sam(clip + ".mp4", a.prompt, n_frames=n, tag=f"eval_sim_drop:{Path(clip).name}")
        Path(clip + ".sam.json").write_text(json.dumps(recs))
        centres = track_centres(recs)
        found = len(centres)
    # image v is measured from the top; z relative to the camera height
    r = measure(centres, gt["fps"], gt["px_per_m"], cy) if found >= 4 else {"g": None}
    if r.get("g") is not None:
        r["z0"] += gt["cam_z"]
        r["err_pct"] = 100 * (r["g"] - gt["g"]) / gt["g"] if gt["g"] else None  # g=None: negative control
    rows.append({"clip": Path(clip).name, "g_true": gt["g"], "frames": n, "tracked": found, "usd": round(usd, 4), **r})
    print(json.dumps(rows[-1]))

if a.out:
    import numpy as np
    neg = [r for r in rows if r["g_true"] is None]
    ok = [r for r in rows if r.get("g") is not None and r["g_true"] is not None]
    err = np.array([abs(r["err_pct"]) for r in ok])
    inside = sum(r["ci95"][0] <= r["g_true"] <= r["ci95"][1] for r in ok)
    summary = {"clips": len(rows), "measured": len(ok), "accepted": sum(bool(r["accepted"]) for r in ok),
               "median_abs_err_pct": float(np.median(err)) if len(ok) else None,
               "p90_abs_err_pct": float(np.percentile(err, 90)) if len(ok) else None,
               "max_abs_err_pct": float(err.max()) if len(ok) else None,
               "ci95_coverage": inside / len(ok) if ok else None,
               "false_accepts_over_2pct": sum(bool(r["accepted"]) and abs(r["err_pct"]) > 2 for r in ok),
               "rejected_over_2pct": sum((not r["accepted"]) and abs(r["err_pct"]) > 2 for r in ok),
               "negatives": len(neg), "negatives_rejected": sum(not r.get("accepted", False) for r in neg),
               "usd": round(sum(r["usd"] for r in rows), 4), "prompt": a.prompt, "oracle": a.oracle}
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary))
print(f"ledger total ${spent():.4f}")
