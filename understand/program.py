"""Understand: turn an observed event list into a task program that generalises to new scenes (Muse Spark)."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apprentice.meta_client import chat

SYSTEM = """You turn one observed demonstration of a tabletop task into a reusable task program.
Infer the most general rule consistent with the demonstration, so the program also works on new scenes with
different positions, different numbers of cubes, and colours not seen in the demonstration.

Program language (JSON only, no prose):
1) a rule applied to every cube:
   {"summary": "...", "for_each": {"object": {"kind": "cube"}, "target": {"kind": "zone", "colour": "SAME_AS_OBJECT"}}}
2) a literal ordered sequence (use when the demo pairs specific colours or builds a structure in an order):
   {"summary": "...", "steps": [{"object": {"colour": "<c>"}, "target": {"kind": "zone" or "cube", "colour": "<c>"}}, ...]}
Use 1) only if every moved cube went to the mat of its own colour. Keep the demonstrated order in 2).
Colours are lowercase single words."""


def understand(observation, model="muse-spark-1.3", effort="minimal", tag="understand"):
    user = json.dumps({k: observation[k] for k in ("scene", "events")}, indent=1)
    text, usd = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                     model=model, effort=effort, max_tokens=3000, tag=tag)
    try:
        return parse_program(text), usd, text
    except (ValueError, AssertionError, KeyError, TypeError) as e:
        raise ValueError(f"invalid program: {e!r}; raw={text!r:.400}") from None


def parse_program(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError(f"no JSON in model output: {text!r:.200}")
    p = json.loads(m.group(0))
    if "for_each" in p:
        t = p["for_each"]["target"]
        assert t["kind"] in ("zone", "cube")
    else:
        for s in p["steps"]:
            assert s["target"]["kind"] in ("zone", "cube") and s["object"]["colour"]
    return p
