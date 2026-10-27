"""Can a frontier multimodal model measure g from video by itself? Same clips, same scale information as Apprentice.

  python scripts/eval_g_direct.py data/simset/c000 ... --model muse-spark-1.3 --out results/g_direct_spark.json
"""
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apprentice import meta_client as mc

Q = ("This {fps} fps video shows a ball moving under gravity in front of a fixed camera (image {W}x{H}, image y grows "
     "downward). Scale: at the ball's distance, 1 metre = {ppm:.1f} pixels. Measure the gravitational acceleration g "
     "in m/s^2 from the ball's motion in this video (do not assume Earth). Reply with JSON only: {{\"g\": number}}")
ap = argparse.ArgumentParser()
ap.add_argument("clips", nargs="+"); ap.add_argument("--model", default="muse-spark-1.3")
ap.add_argument("--effort", default="high"); ap.add_argument("--out")
a = ap.parse_args()
pin, pout = mc.PRICES[a.model]
rows = []
for clip in a.clips:
    gt = json.load(open(clip + ".json"))
    mc._check_budget(0.2)
    body = {"model": a.model, "max_output_tokens": 12000, "reasoning": {"effort": a.effort},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": Q.format(fps=gt["fps"], W=gt["W"], H=gt["H"], ppm=gt["px_per_m"])},
                {"type": "input_video", "video_url": mc._data_url(clip + ".mp4")}]}]}
    d = mc._post("/responses", body, timeout=300)
    u = d.get("usage", {})
    usd = u.get("input_tokens", 0) * pin / 1e6 + u.get("output_tokens", 0) * pout / 1e6
    mc._log(a.model, usd, u, f"g_direct:{Path(clip).name}")
    text = "".join(c.get("text", "") for o in d.get("output", []) if o.get("type") == "message" for c in o.get("content", []))
    m = re.search(r'"g"\s*:\s*(-?[0-9.]+)', text)
    g = float(m.group(1)) if m else None
    rows.append({"clip": Path(clip).name, "g_true": gt["g"], "g_pred": g, "usd": usd,
                 "err_pct": 100 * (g - gt["g"]) / gt["g"] if g else None})
    print(json.dumps(rows[-1]))
ok = [r for r in rows if r["g_pred"] is not None]
errs = sorted(abs(r["err_pct"]) for r in ok)
summary = {"model": a.model, "clips": len(rows), "answered": len(ok),
           "median_abs_err_pct": errs[len(errs) // 2] if errs else None,
           "said_9.81": sum(abs(r["g_pred"] - 9.81) < 0.05 for r in ok),
           "usd": round(sum(r["usd"] for r in rows), 4)}
print(json.dumps(summary))
if a.out:
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
