# Apprentice

An AI that learns physical work by watching: SAM 3.1 tracks what happens in a video, Muse models turn the tracks into
measured physics and a step-by-step procedure, a MuJoCo robot executes it and checks itself with the same eyes,
and a coach mode flags the step a person got wrong. Every number is checked against physics or re-observation,
every model call is priced.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026). All code in this repo was written during
the event.

## Build log
- **2026-10-27** — Meta Model API client with a local spend ledger and hard budget stop (`apprentice/meta_client.py`).
  MuJoCo free-fall scenes with exact camera intrinsics and ground truth (`sim/drop_scene.py`).
  Measure mode v0: SAM 3.1 video track → metres → constant-acceleration fit with residual-bootstrap 95% CI
  (`measure/fit.py`, `scripts/eval_sim_drop.py`).
  First results (SAM 3.1, prompt "red ball", 60 frames @ 60 fps, 640×480):

  | clip | g true | g measured | 95% CI | error |
  |---|---|---|---|---|
  | Earth sim | 9.81 | 9.841 | 9.813–9.868 | +0.31% |
  | Moon sim | 1.62 | 1.6205 | 1.6149–1.6262 | +0.03% |

  Fit code checked first on true positions (error 0 after fixing a frame-timing bug in the simulator: 33 × 0.5 ms
  physics steps per 1/60 s frame had made every clip 2% slow).
  Apollo 15 hammer–feather (NASA NSSDC footage): Muse Voice Transcribe turns located the drop between 58.9 s and
  61.4 s. SAM 3.1 text prompts "hammer"/"feather" returned no objects, neither on the original 720×480 clip nor on a
  3× upscaled crop — the objects are a few pixels wide and low contrast against the white suit. Next: point prompts
  and a time-of-flight estimate with a size reference.

## Run
```
python3 -m venv .venv && .venv/bin/pip install mujoco numpy scipy pillow imageio imageio-ffmpeg
echo 'META_API_KEY=...' > ~/.config/meta/keys.env && chmod 600 ~/.config/meta/keys.env
.venv/bin/python sim/drop_scene.py data/sim/moon --g 1.62
.venv/bin/python scripts/eval_sim_drop.py data/sim/moon --oracle   # no API cost
.venv/bin/python scripts/eval_sim_drop.py data/sim/moon            # SAM 3.1, ~$0.012
```
Budget cap: `APPRENTICE_BUDGET_USD` (default 10), enforced from `runs/ledger.jsonl` before every call.
