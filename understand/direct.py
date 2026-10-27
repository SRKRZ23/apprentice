"""Baseline Understand without SAM: give the raw demonstration video to a multimodal Muse model and ask for the
task program directly. Used to measure what the SAM 3.1 tracking stage adds (accuracy and cost)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apprentice import meta_client as mc
from understand.program import SYSTEM, parse_program


def understand_direct(video, model="muse-spark-1.3", effort="low", max_tokens=6000, tag="direct"):
    pin, pout = mc.PRICES[model]
    mc._check_budget(0.05)
    body = {"model": model, "max_output_tokens": max_tokens, "reasoning": {"effort": effort},
            "instructions": SYSTEM,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "This video shows one demonstration of a tabletop task by a robot "
                                               "gripper. Write the task program."},
                {"type": "input_video", "video_url": mc._data_url(video)}]}]}
    d = mc._post("/responses", body, timeout=300)
    u = d.get("usage", {})
    usd = u.get("input_tokens", 0) * pin / 1e6 + u.get("output_tokens", 0) * pout / 1e6
    mc._log(model, usd, u, tag)
    text = "".join(c.get("text", "") for o in d.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []))
    try:
        return parse_program(text), usd, text, u
    except Exception as e:
        return None, usd, text, u
