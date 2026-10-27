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


def free_flight_window(v):
    """Indices up to the lowest point before the first bounce (image v grows downward)."""
    v = np.asarray(v)
    k = int(np.argmax(v))
    # if the object bounced, v rises again after k; cut 2 frames before the minimum height to avoid contact
    return slice(0, max(k - 1, 3)) if k < len(v) - 1 else slice(0, len(v))


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
    w = free_flight_window(v)
    t = np.array(frames, float)[w] / fps
    z = (cy - v[w]) / px_per_m
    out = fit_g(t, z)
    out["accepted"] = out["rms_m"] < 2.0 / px_per_m  # residuals within ~2 px
    out["frames_used"] = (frames[w][0], frames[w][-1]) if len(frames[w]) else None
    return out
