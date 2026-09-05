# labpilot/journal.py — what the labpilot actually did, and what it cost.
#
# You cannot improve what you do not measure. Without this there is no way to
# say which stage is slow, what a cycle costs, or whether the failure rate is
# getting better or worse.
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parent / "memory" / "journal.jsonl"


def write(event: str, **fields) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"t": datetime.now().isoformat(timespec="seconds"), "event": event}
    rec.update(fields)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


@contextmanager
def stage(name: str, **fields):
    """Time a stage and record it whether it succeeds or fails."""
    t0 = time.time()
    write(f"{name}.start", **fields)
    try:
        yield
    except BaseException as exc:
        write(f"{name}.fail", seconds=round(time.time() - t0, 1),
              error=f"{type(exc).__name__}: {exc}"[:200], **fields)
        raise
    else:
        write(f"{name}.ok", seconds=round(time.time() - t0, 1), **fields)


def entries() -> list[dict]:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def report() -> str:
    """What has this thing cost, and where does the time go?"""
    ev = entries()
    if not ev:
        return "no activity recorded yet."

    stages: dict[str, list] = {}
    fails: dict[str, int] = {}
    for e in ev:
        name, _, kind = e["event"].rpartition(".")
        if kind == "ok" and e.get("seconds") is not None:
            stages.setdefault(name, []).append(e["seconds"])
        elif kind == "fail":
            fails[name] = fails.get(name, 0) + 1

    L = ["", "WHERE THE TIME GOES", ""]
    L.append(f"  {'stage':16}{'runs':>6}{'total':>10}{'median':>9}{'fails':>7}")
    total = 0.0
    for name, secs in sorted(stages.items(), key=lambda kv: -sum(kv[1])):
        s = sorted(secs)
        med = s[len(s) // 2]
        total += sum(secs)
        L.append(f"  {name:16}{len(secs):>6}{sum(secs)/60:>9.1f}m{med:>8.1f}s"
                 f"{fails.get(name, 0):>7}")
    L.append(f"  {'TOTAL':16}{'':>6}{total/60:>9.1f}m")

    gpu = [e for e in ev if e["event"].startswith("gpu.")]
    gpu_min = sum(e.get("seconds", 0) for e in gpu if e["event"] == "gpu.ok") / 60
    blocked = len([e for e in ev if e["event"] == "gate.blocked"])
    approved = len([e for e in ev if e["event"] == "gate.approved"])
    L += ["", "GPU ECONOMY", "",
          f"  GPU minutes spent      {gpu_min:.1f}",
          f"  runs approved          {approved}",
          f"  runs blocked (saved)   {blocked}"]
    if approved + blocked:
        L.append(f"  blocked fraction       {blocked/(approved+blocked):.0%}")

    gen = [e for e in ev if e["event"].startswith("design.attempt")]
    ok = len([e for e in gen if e.get("verified")])
    if gen:
        L += ["", "CODE GENERATION", "",
              f"  attempts               {len(gen)}",
              f"  verified first try     {len([e for e in gen if e.get('attempt')==1 and e.get('verified')])}",
              f"  overall success rate   {ok/len(gen):.0%}"]
    return "\n".join(L)
