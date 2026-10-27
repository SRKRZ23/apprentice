"""Measure: fit constant-acceleration motion to a tracked object and report g with uncertainty.

Track -> vertical image coordinate per frame -> metres (via px_per_m from calibration or a reference object)
-> fit z(t) = z0 + v0 t - g t^2 / 2 on the free-flight window (frames before the first impact)
-> residual bootstrap for a 95% interval. Acceptance: residual RMS below a pixel-scale threshold.
"""
import numpy as np


def track_centres(records, pick="largest-motion"):
    """records from meta_client.sam -> {frame: (u, v)} for one object id.

    Default: the id whose box centre moves the most vertically (the falling object, not shadows/walls).
    """
    by_id = {}
    for r in records:
        x1, y1, x2, y2 = r["box"]
        by_id.setdefault(r["id"], {})[r["frame"]] = ((x1 + x2) / 2, (y1 + y2) / 2)
    if not by_id:
        return {}
    span = {i: np.ptp([v for _, v in t.values()]) if len(t) > 1 else 0 for i, t in by_id.items()}
    return by_id[max(span, key=span.get)]


def free_flight_window(v, t, start=8):
    """Frames before the first impact (image v grows downward).

    Grow the window one frame at a time; impact = first frame k where frames k and k+1 both sit more than 3 sigma
    above (higher in the world than) the parabola fitted to frames [0, k) -- a bounce or the object at rest.
    Speed- or argmax-based rules fail on noisy video: jitter looks like a stop, a resting object looks like motion.
    """
    v, t = np.asarray(v, float), np.asarray(t, float)
    A = np.stack([np.ones_like(t), t, t ** 2], 1)
    for k in range(start, len(v) - 1):
        c, *_ = np.linalg.lstsq(A[:k], v[:k], rcond=None)
        sigma = max(np.sqrt(np.mean((v[:k] - A[:k] @ c) ** 2)), 0.5)
        short = A[k:k + 2] @ c - v[k:k + 2]
        if (short > 3 * sigma).all():
            return slice(0, k - 1)  # drop the last frame before impact (contact, partial occlusion)
    return slice(0, len(v))


def fit_g(t, z, n_boot=2000, seed=0):
    t, z = np.asarray(t, float), np.asarray(z, float)
    A = np.stack([np.ones_like(t), t, -0.5 * t ** 2], 1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    res = z - A @ coef
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        zb = A @ coef + rng.choice(res, len(res), replace=True)
        boots.append(np.linalg.lstsq(A, zb, rcond=None)[0][2])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"g": float(coef[2]), "ci95": (float(lo), float(hi)), "v0": float(coef[1]), "z0": float(coef[0]),
            "rms_m": float(np.sqrt(np.mean(res ** 2))), "n": int(len(t))}


def measure(centres, fps, px_per_m, cy):
    frames = sorted(centres)
    v = np.array([centres[f][1] for f in frames])
    w = free_flight_window(v, np.array(frames, float) / fps)
    t = np.array(frames, float)[w] / fps
    z = (cy - v[w]) / px_per_m
    out = fit_g(t, z)
    # accept only if residuals are pixel-sized AND the trajectory end is not systematically off the parabola
    A = np.stack([np.ones_like(t), t, -0.5 * t ** 2], 1)
    res = z - A @ np.array([out["z0"], out["v0"], out["g"]])
    tail = abs(res[-5:].mean()) if len(res) >= 8 else 0.0
    out["tail_bias_px"] = float(tail * px_per_m)
    out["accepted"] = bool(out["n"] >= 8 and out["rms_m"] < 2.0 / px_per_m and tail * px_per_m < 1.0)
    out["frames_used"] = (frames[w][0], frames[w][-1]) if len(frames[w]) else None
    return out
