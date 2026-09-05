# labpilot/checkpoint.py — a long cycle should survive an SSH drop.
#
# Discovery costs minutes of LLM time and design costs more. Losing that to a
# dropped connection and starting over is the most annoying possible failure,
# and the cheapest to prevent.
from __future__ import annotations

import json
from pathlib import Path

CKPT = Path(__file__).resolve().parent / "work" / "cycle.json"


def save(stage: str, **data) -> None:
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    cur = load()
    cur["stage"] = stage
    cur.setdefault("done", [])
    if stage not in cur["done"]:
        cur["done"].append(stage)
    cur.update(data)
    CKPT.write_text(json.dumps(cur, indent=2, default=str), encoding="utf-8")


def load() -> dict:
    if not CKPT.exists():
        return {}
    try:
        return json.loads(CKPT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def done(stage: str) -> bool:
    return stage in load().get("done", [])


def clear() -> None:
    CKPT.unlink(missing_ok=True)


def describe() -> str:
    c = load()
    if not c:
        return "no cycle in progress."
    return (f"cycle in progress · last stage '{c.get('stage')}' · "
            f"completed {c.get('done')} · resume with `cycle --resume`")
