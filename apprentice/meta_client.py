"""Meta Model API client with a local spend ledger and a hard budget stop.

Every call is priced from the usage the API returns and appended to runs/ledger.jsonl.
Before each call the ledger total is checked against APPRENTICE_BUDGET_USD (default 10);
the Billing page only sends notifications, so this is the actual cap.
"""
import base64, json, os, re, time
from pathlib import Path
import urllib.request, urllib.error

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "runs" / "ledger.jsonl"
BASE = "https://api.meta.ai/v1"

# USD. Text: per 1M tokens (input, output). SAM: per image / per video frame (50-frame minimum per video request).
PRICES = {
    "muse-spark-1.3": (1.25, 4.25), "muse-spark-1.2": (1.25, 4.25), "muse-spark-1.1": (1.25, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.20), "muse-spark-1.2-contributor": (0.10, 0.20),
    "muse-wonder-1.0": (4.0, 20.0),
}
SAM_IMAGE = 2.50 / 1000
SAM_FRAME = 0.20 / 1000


class BudgetExceeded(RuntimeError):
    pass


def _key():
    k = os.environ.get("META_API_KEY")
    if not k:
        for line in (Path.home() / ".config/meta/keys.env").read_text().splitlines():
            if line.startswith("META_API_KEY="):
                k = line.split("=", 1)[1].strip()
    return k


def spent():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def _check_budget(estimate=0.0):
    budget = float(os.environ.get("APPRENTICE_BUDGET_USD", "10"))
    if spent() + estimate > budget:
        raise BudgetExceeded(f"ledger ${spent():.4f} + est ${estimate:.4f} > budget ${budget}")


def _log(model, usd, usage, tag):
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"t": time.time(), "model": model, "usd": usd, "usage": usage, "tag": tag}) + "\n")


def _post(path, body, timeout=120, attempts=4):
    """POST with retries on timeouts, 429 and 5xx (exponential backoff). A timed-out request may still have been
    billed by the server; such attempts are counted in runs/timeouts.jsonl so the ledger gap is visible."""
    data = json.dumps(body).encode()
    for i in range(attempts):
        req = urllib.request.Request(BASE + path, data=data, method="POST",
                                     headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or i == attempts - 1:
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:500]}") from None
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            with (LEDGER.parent / "timeouts.jsonl").open("a") as f:
                f.write(json.dumps({"t": time.time(), "path": path, "model": body.get("model"), "err": repr(e)}) + "\n")
            if i == attempts - 1:
                raise
        time.sleep(2 ** i * 3)


def chat(messages, model="muse-spark-1.3", effort="minimal", max_tokens=2000, tag="", **kw):
    """Chat Completions. Reasoning shares max_tokens with the answer, so effort is always explicit."""
    pin, pout = PRICES[model]
    _check_budget(max_tokens * pout / 1e6)
    d = _post("/chat/completions", {"model": model, "messages": messages, "max_tokens": max_tokens,
                                    "reasoning_effort": effort, **kw})
    u = d["usage"]
    usd = u["prompt_tokens"] * pin / 1e6 + u["completion_tokens"] * pout / 1e6
    _log(model, usd, u, tag)
    content = d["choices"][0]["message"]["content"]
    if content is None and d["choices"][0]["finish_reason"] == "length" and max_tokens < 16000:
        # reasoning used the whole budget before any answer: retry once with twice the room
        content, more = chat(messages, model, effort, max_tokens * 2, tag + ":retry", **kw)
        usd += more
    return content, usd


def _data_url(path):
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".mp4": "video/mp4", ".mov": "video/quicktime"}[Path(path).suffix.lower()]
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def sam(media_path, prompt, n_frames=None, tag="", confidence=True):
    """SAM 3.1 on an image or a video with one noun-phrase concept. Returns (records, usd).

    records: list of {frame, id, box: (x1, y1, x2, y2), w, h, c}. Masks are kept raw in 'mask'.
    """
    is_video = Path(media_path).suffix.lower() in (".mp4", ".mov")
    est = max(n_frames or 50, 50) * SAM_FRAME if is_video else SAM_IMAGE
    _check_budget(est)
    part = ({"type": "input_video", "video_url": _data_url(media_path)} if is_video
            else {"type": "input_image", "image_url": _data_url(media_path)})
    d = _post("/responses", {"model": "sam-3.1", "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": prompt}, part]}],
        "metadata": {"include_confidence": str(confidence).lower(), "mask_encoding": "one_bit"}})
    text = "".join(c.get("text", "") for o in d.get("output", []) for c in o.get("content", []))
    u = d.get("usage", {})
    if is_video:
        frames = u.get("frames_processed") or u.get("video_frames_processed") or n_frames or 50
        usd = max(frames, 50) * SAM_FRAME
    else:
        usd = u.get("images_processed", 1) * SAM_IMAGE
    _log("sam-3.1", usd, u, tag)
    return parse_sam(text), usd


_REC = re.compile(r"(?:<(\d+)f>)?(\d+)<\|box;([^|]*)\|>(?:<\|mask;([^|]*)\|>)?")


def parse_sam(text):
    """Parse '<Nf>id<|box;x1=..;..|><|mask;..|>,id<|box..|>...' into records (frame marker carries over)."""
    out, frame = [], 0
    for line in text.splitlines() or [text]:
        for m in _REC.finditer(line):
            if m.group(1) is not None:
                frame = int(m.group(1))
            kv = dict(p.split("=", 1) for p in m.group(3).split(";") if "=" in p)
            out.append({"frame": frame, "id": int(m.group(2)),
                        "box": tuple(float(kv[k]) for k in ("x1", "y1", "x2", "y2")),
                        "w": int(kv["w"]), "h": int(kv["h"]), "c": float(kv.get("c", "nan")),
                        "mask": m.group(4)})
    return out
